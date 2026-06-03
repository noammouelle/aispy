"""
Phase space map surrogate for ais++ simulations.

Loads a _PSMAP.h5 file produced by ais++ in ``psgrid`` mode, builds
per-port interpolators for the wavepacket amplitudes and phase shifts,
and generates synthetic atom observations for arbitrary Gaussian clouds
and phase profiles — without re-running ais++.

GPU acceleration via CuPy
--------------------------
When CuPy is available the **entire** ``generate_atoms`` pipeline runs on
the GPU:

  1. (CPU) Sample (x0, vx0) with numpy — stays on CPU as scipy input.
  2. (GPU) Bilinear interpolation on the regular PSMAP grid.  Because the
     grid is regular (linspace) the index computation reduces to one
     division per dimension; no scipy call is made in this path.
  3. (GPU) Port probabilities via _port_prob (cos, multiply, reduce).
  4. (GPU) Bernoulli sampling with CuPy's PRNG.
  5. (CPU) Thin host←device copy for the output DataFrame.

The only CPU work inside the hot path is sampling (x0, vx0) and
evaluating the user-supplied ``phase_profile``.  scipy interpolation is
used only when ``use_gpu=False`` or CuPy is unavailable.

Typical usage
-------------
>>> from aispy.psmap import load_psmap, PSMAPSurrogate
>>> sur = PSMAPSurrogate(load_psmap('PSR_EXAMPLE_PSGRID.h5'), t_det=4.451)
>>> df  = sur.generate_atoms(mu_x0=0., mu_vx0=0.,
...                          sigma_x=1e-4, sigma_vx=3.1e-4,
...                          phi0=0., natoms=500_000,
...                          phase_profile=lambda xf, vxf: 3140. * xf)
"""

import warnings
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def _get_xp(arr):
    """Return the array module (cupy or numpy) matching *arr*."""
    if _CUPY_AVAILABLE:
        return cp.get_array_module(arr)
    return np


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_psmap(fname):
    """
    Load a PSMAP HDF5 file into a plain dict of numpy arrays.

    Parameters
    ----------
    fname : str or Path

    Returns
    -------
    dict with keys matching the datasets written by
    AISDriver::WritePhaseSpaceMapToFile.
    """
    with h5py.File(fname, 'r') as f:
        decode = lambda arr: np.array(
            [p.decode() if isinstance(p, bytes) else p for p in arr])
        return {
            'initial_positions':  f['initial_positions'][:],
            'initial_velocities': f['initial_velocities'][:],
            'final_positions':    f['final_positions'][:],
            'final_velocities':   f['final_velocities'][:],
            'phase_shifts':       f['phase_shifts'][:],
            'phase_shift_errors': f['phase_shift_errors'][:],
            'amp0':               f['amp0'][:],
            'amp1':               f['amp1'][:],
            'states':             f['states'][:],
            'is_interfering':     f['is_interfering'][:],
            'atom_indices':       f['atom_indices'][:],
            'path0':              decode(f['path0'][:]),
            'path1':              decode(f['path1'][:]),
        }


# ── Core probability formula ──────────────────────────────────────────────────

def _port_prob(amp0, amp1, dphi, is_interfering, delta=0.0):
    """
    Port probability P = A0² + A1² + interfering·2A0A1·cos(Δφ + δ).

    Works transparently with both numpy and cupy arrays via
    ``cp.get_array_module``.
    """
    xp = _get_xp(amp0)
    return (amp0**2 + amp1**2
            + is_interfering * 2.0 * amp0 * amp1 * xp.cos(dphi + delta))


# ── GPU bilinear interpolation on a regular grid ──────────────────────────────

def _find_cell(arr, lo, dx, n):
    """
    For a uniform grid with spacing *dx* starting at *lo*, return the
    lower cell index and fractional offset for each element of *arr*.
    Works with both numpy and cupy arrays.
    """
    xp  = _get_xp(arr)
    raw = (arr - lo) / dx
    idx = xp.clip(raw.astype(xp.int32), 0, n - 2)
    tx  = xp.clip(raw - idx.astype(xp.float64), 0.0, 1.0)
    return idx, tx


def _bilinear(grid, ix, ivx, tx, tvx):
    """
    Bilinear interpolation on a 2D grid using pre-computed cell indices
    and fractional offsets.  All arrays may be cupy or numpy.
    """
    f00 = grid[ix,   ivx  ]
    f10 = grid[ix+1, ivx  ]
    f01 = grid[ix,   ivx+1]
    f11 = grid[ix+1, ivx+1]
    return (f00*(1-tx)*(1-tvx) + f10*tx*(1-tvx)
          + f01*(1-tx)*tvx    + f11*tx*tvx)


# ── Surrogate ─────────────────────────────────────────────────────────────────

class PSMAPSurrogate:
    """
    Surrogate model built from an ais++ phase space map.

    The PSMAP stores, for every node (x0, vx0) on a regular grid and every
    output port, the wavepacket amplitudes and the phase difference Δφ
    (computed in quad precision inside ais++).

    GPU acceleration
    ----------------
    When CuPy is available and ``use_gpu=True`` (the default), the residual
    grids are uploaded to the GPU at construction time and the **entire**
    ``generate_atoms`` pipeline — interpolation, probability computation, and
    Bernoulli sampling — runs on the GPU.  The CPU overhead per call is
    limited to sampling (x0, vx0) with the numpy PRNG and a thin
    host←device copy for the output.

    Parameters
    ----------
    psmap : dict
        Output of :func:`load_psmap`.
    t_det : float
        Detection time [s].
    use_gpu : bool
        Use CuPy when available (default ``True``).
    """

    def __init__(self, psmap, t_det, use_gpu=True):
        self.t_det = float(t_det)

        self._use_gpu = use_gpu and _CUPY_AVAILABLE
        if use_gpu and not _CUPY_AVAILABLE:
            warnings.warn('CuPy is not installed — falling back to NumPy.',
                          stacklevel=2)

        atom_idx = psmap['atom_indices']
        n_atoms  = int(atom_idx.max()) + 1
        nP       = int(np.bincount(atom_idx)[0])
        self.nP  = nP

        # Identify regular (x0, vx0) grid
        first     = np.searchsorted(atom_idx, np.arange(n_atoms))
        x0_atoms  = psmap['initial_positions'][first, 0]
        vx0_atoms = psmap['initial_velocities'][first, 0]
        self.xs   = np.unique(x0_atoms)
        self.vxs  = np.unique(vx0_atoms)
        self.nx   = len(self.xs)
        self.nvx  = len(self.vxs)

        # Grid spacing (uniform by construction from ais++ linspace)
        self._x_lo  = float(self.xs[0])
        self._vx_lo = float(self.vxs[0])
        self._dx    = float((self.xs[-1]  - self.xs[0])  / (self.nx  - 1)) if self.nx  > 1 else 1.0
        self._dvx   = float((self.vxs[-1] - self.vxs[0]) / (self.nvx - 1)) if self.nvx > 1 else 1.0

        def to_grid(a):
            return a.reshape(n_atoms, nP).reshape(self.nx, self.nvx, nP)

        dphi  = to_grid(psmap['phase_shifts'])
        amp0  = to_grid(psmap['amp0'])
        amp1  = to_grid(psmap['amp1'])
        inter = to_grid(psmap['is_interfering'].astype(float))
        state = to_grid(psmap['states'])

        self.port_states      = state[0, 0, :]
        self.port_interfering = inter[0, 0, :]
        self.port_path0 = psmap['path0'][first[0]:first[0]+nP]
        self.port_path1 = psmap['path1'][first[0]:first[0]+nP]

        # Decompose dphi into a linear trend (stripped before interpolation)
        # and a smooth residual that the interpolator actually sees.
        X0g  = self.xs[:, None]  * np.ones((1, self.nvx))
        VX0g = np.ones((self.nx, 1)) * self.vxs[None, :]
        self._dphi_linear = np.zeros((nP, 3))
        dphi_resid = np.empty_like(dphi)

        for pi in range(nP):
            d = dphi[:, :, pi]
            A = np.column_stack([np.ones(self.nx * self.nvx),
                                 X0g.ravel(), VX0g.ravel()])
            c, *_ = np.linalg.lstsq(A, d.ravel(), rcond=None)
            self._dphi_linear[pi] = c
            dphi_resid[:, :, pi]  = d - (c[0] + c[1]*X0g + c[2]*VX0g)

        # CPU interpolators (used when use_gpu=False)
        kw = dict(method='linear', bounds_error=False, fill_value=None)
        self._interp_dphi = [
            RegularGridInterpolator((self.xs, self.vxs), dphi_resid[:,:,pi], **kw)
            for pi in range(nP)]
        self._interp_amp0 = [
            RegularGridInterpolator((self.xs, self.vxs), amp0[:,:,pi], **kw)
            for pi in range(nP)]
        self._interp_amp1 = [
            RegularGridInterpolator((self.xs, self.vxs), amp1[:,:,pi], **kw)
            for pi in range(nP)]

        # GPU: upload grids once at init time
        if self._use_gpu:
            self._gpu_dphi_resid = [cp.asarray(dphi_resid[:,:,pi], dtype=cp.float64)
                                    for pi in range(nP)]
            self._gpu_amp0 = [cp.asarray(amp0[:,:,pi], dtype=cp.float64)
                              for pi in range(nP)]
            self._gpu_amp1 = [cp.asarray(amp1[:,:,pi], dtype=cp.float64)
                              for pi in range(nP)]
            self._port_states_gpu = cp.asarray(self.port_states)
            self._port_inter_gpu  = cp.asarray(self.port_interfering)

        self._fit_metrics = self._compute_fit_metrics(dphi, X0g, VX0g)

    # ── Fit quality ──────────────────────────────────────────────────────────

    def _compute_fit_metrics(self, dphi, X0g, VX0g):
        x0f  = X0g.ravel(); vx0f = VX0g.ravel()
        A = np.column_stack([np.ones_like(x0f), x0f, vx0f,
                             x0f**2, x0f*vx0f, vx0f**2])
        metrics = {}
        for pi in range(self.nP):
            d      = dphi[:, :, pi].ravel()
            c, *_  = np.linalg.lstsq(A, d, rcond=None)
            pred   = A @ c
            ss_res = np.sum((d - pred)**2)
            ss_tot = np.sum((d - d.mean())**2)
            metrics[pi] = {
                'r2':               float(1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0),
                'residual_std_rad': float(np.sqrt(ss_res / len(d))),
                'quad_coeffs':      c,
            }
        return metrics

    def fit_quality(self):
        """
        Return quality metrics for a quadratic fit to Δφ(x0, vx0).

        Returns
        -------
        dict keyed by port index with sub-keys ``r2`` and
        ``residual_std_rad``.
        """
        return {pi: {k: v for k, v in m.items() if k != 'quad_coeffs'}
                for pi, m in self._fit_metrics.items()}

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _eval_gpu(self, x0_gpu, vx0_gpu):
        """Evaluate on GPU using bilinear interpolation on the regular grid."""
        ix,  tx  = _find_cell(x0_gpu,  self._x_lo,  self._dx,  self.nx)
        ivx, tvx = _find_cell(vx0_gpu, self._vx_lo, self._dvx, self.nvx)

        dphi_out = cp.empty((len(x0_gpu), self.nP), dtype=cp.float64)
        amp0_out = cp.empty((len(x0_gpu), self.nP), dtype=cp.float64)
        amp1_out = cp.empty((len(x0_gpu), self.nP), dtype=cp.float64)

        for pi in range(self.nP):
            c = self._dphi_linear[pi]
            dphi_out[:, pi] = (_bilinear(self._gpu_dphi_resid[pi], ix, ivx, tx, tvx)
                               + c[0] + c[1]*x0_gpu + c[2]*vx0_gpu)
            amp0_out[:, pi] = _bilinear(self._gpu_amp0[pi], ix, ivx, tx, tvx)
            amp1_out[:, pi] = _bilinear(self._gpu_amp1[pi], ix, ivx, tx, tvx)

        return dphi_out, amp0_out, amp1_out

    def eval(self, x0_arr, vx0_arr):
        """
        Evaluate the surrogate at arbitrary (x0, vx0) coordinates.

        Uses GPU bilinear interpolation when ``use_gpu=True``, scipy on CPU
        otherwise.  Always returns **numpy** arrays so the result can be
        used directly with scipy or pandas.

        Parameters
        ----------
        x0_arr, vx0_arr : array-like, shape (N,)

        Returns
        -------
        dphi  : ndarray (N, nP)
        amp0  : ndarray (N, nP)
        amp1  : ndarray (N, nP)
        """
        x0_arr  = np.asarray(x0_arr,  dtype=float)
        vx0_arr = np.asarray(vx0_arr, dtype=float)

        out_frac = (
            np.mean((x0_arr  < self.xs[0])  | (x0_arr  > self.xs[-1]))
          + np.mean((vx0_arr < self.vxs[0]) | (vx0_arr > self.vxs[-1]))
        ) / 2.0
        if out_frac > 0.01:
            warnings.warn(
                f'{out_frac*100:.1f}% of sampled atoms lie outside the PSMAP '
                f'grid (x0 ∈ [{self.xs[0]*1e6:.0f}, {self.xs[-1]*1e6:.0f}] µm, '
                f'vx0 ∈ [{self.vxs[0]*1e3:.2f}, {self.vxs[-1]*1e3:.2f}] mm/s). '
                'Regenerate the PSMAP with a wider grid.',
                stacklevel=3)

        if self._use_gpu:
            x0_gpu  = cp.asarray(x0_arr)
            vx0_gpu = cp.asarray(vx0_arr)
            dphi_g, amp0_g, amp1_g = self._eval_gpu(x0_gpu, vx0_gpu)
            return cp.asnumpy(dphi_g), cp.asnumpy(amp0_g), cp.asnumpy(amp1_g)

        pts      = np.column_stack([x0_arr, vx0_arr])
        dphi_out = np.empty((len(x0_arr), self.nP))
        amp0_out = np.empty((len(x0_arr), self.nP))
        amp1_out = np.empty((len(x0_arr), self.nP))
        for pi in range(self.nP):
            c = self._dphi_linear[pi]
            dphi_out[:, pi] = (self._interp_dphi[pi](pts)
                               + c[0] + c[1]*x0_arr + c[2]*vx0_arr)
            amp0_out[:, pi] = self._interp_amp0[pi](pts)
            amp1_out[:, pi] = self._interp_amp1[pi](pts)
        return dphi_out, amp0_out, amp1_out

    # ── Atom generation ───────────────────────────────────────────────────────

    def generate_atoms(self, mu_x0, mu_vx0, sigma_x, sigma_vx,
                       phi0=0.0, natoms=100_000,
                       phase_profile=None, rng=None):
        """
        Sample atoms from a Gaussian cloud and return synthetic observations.

        GPU pipeline (when CuPy is available)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1. CPU  — sample (x0, vx0); compute xf = x0 + vx0·t_det.
        2. CPU  — evaluate phase_profile(xf, vxf) if provided.
        3. GPU  — host→device transfer of x0, vx0, delta (3N float64).
        4. GPU  — bilinear interpolation of dphi, amp0, amp1.
        5. GPU  — _port_prob and Bernoulli sampling.
        6. CPU  — device→host transfer of state, prob_s0 (N · 9 bytes).

        The ``phase_profile`` callable receives **numpy** arrays and must
        return an array-like; its output is moved to the GPU automatically.

        Parameters
        ----------
        mu_x0, mu_vx0     : float   Cloud centre-of-mass [m, m/s]
        sigma_x, sigma_vx  : float  Cloud 1σ spreads [m, m/s]
        phi0               : float  Global interferometer phase offset [rad]
        natoms             : int    Number of atoms to simulate
        phase_profile      : callable or None
            ``f(xf, vxf) -> array (natoms,)`` [rad].
            Example — phase shear: ``lambda xf, vxf: kappa * xf``
        rng                : numpy Generator, int seed, or None

        Returns
        -------
        pandas.DataFrame with columns:
            x0, vx0   Initial phase-space coordinates [m, m/s]
            xf, vxf   Final transverse position and velocity [m, m/s]
            state     Output internal state (0 = ground, 1 = excited)
            prob_s0   Probability of ground state for this atom
        """
        import pandas as pd

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # ── 1. Sample + kinematic map (CPU) ──────────────────────────────────
        x0  = rng.normal(mu_x0,  sigma_x,  natoms)
        vx0 = rng.normal(mu_vx0, sigma_vx, natoms)
        xf  = x0 + vx0 * self.t_det
        vxf = vx0.copy()

        # ── 2. Phase profile (CPU — user callable receives numpy) ─────────────
        delta_np = (np.asarray(phase_profile(xf, vxf), dtype=np.float64)
                    if phase_profile is not None
                    else np.zeros(natoms, dtype=np.float64))

        if self._use_gpu:
            # ── 3. Host → device ─────────────────────────────────────────────
            x0_gpu    = cp.asarray(x0,       dtype=cp.float64)
            vx0_gpu   = cp.asarray(vx0,      dtype=cp.float64)
            delta_gpu = cp.asarray(delta_np, dtype=cp.float64)

            # ── 4. GPU bilinear interpolation ─────────────────────────────────
            dphi_g, amp0_g, amp1_g = self._eval_gpu(x0_gpu, vx0_gpu)

            # ── 5. Port probabilities + Bernoulli sampling ────────────────────
            total_phase = dphi_g + (delta_gpu + phi0)[:, None]
            probs   = _port_prob(amp0_g, amp1_g, dphi_g,
                                 self._port_inter_gpu[None, :],
                                 delta=total_phase - dphi_g)
            s0_mask = (self._port_states_gpu == 0)
            prob_s0 = cp.clip(probs[:, s0_mask].sum(axis=1), 0.0, 1.0)
            state   = (cp.random.random(natoms, dtype=cp.float64) > prob_s0).astype(cp.int8)

            # ── 6. Device → host ─────────────────────────────────────────────
            prob_s0_np = cp.asnumpy(prob_s0)
            state_np   = cp.asnumpy(state)

        else:
            pts    = np.column_stack([x0, vx0])
            dphi_np = np.empty((natoms, self.nP))
            amp0_np = np.empty((natoms, self.nP))
            amp1_np = np.empty((natoms, self.nP))
            for pi in range(self.nP):
                c = self._dphi_linear[pi]
                dphi_np[:, pi] = (self._interp_dphi[pi](pts)
                                  + c[0] + c[1]*x0 + c[2]*vx0)
                amp0_np[:, pi] = self._interp_amp0[pi](pts)
                amp1_np[:, pi] = self._interp_amp1[pi](pts)

            total_phase = dphi_np + (delta_np + phi0)[:, None]
            probs   = _port_prob(amp0_np, amp1_np, dphi_np,
                                 self.port_interfering[None, :],
                                 delta=total_phase - dphi_np)
            s0_mask    = (self.port_states == 0)
            prob_s0_np = np.clip(probs[:, s0_mask].sum(axis=1), 0.0, 1.0)
            state_np   = (rng.uniform(size=natoms) > prob_s0_np).astype(np.int8)

        return pd.DataFrame({'x0': x0, 'vx0': vx0,
                             'xf': xf, 'vxf': vxf,
                             'state': state_np, 'prob_s0': prob_s0_np})
