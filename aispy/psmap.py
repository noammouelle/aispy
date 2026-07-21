"""
Phase space map surrogate for ais++ simulations.

Loads a _PSMAP.h5 file produced by ais++ in ``psgrid`` mode, builds
per-port interpolators for the wavepacket amplitudes and phase shifts,
and generates synthetic atom observations for arbitrary Gaussian clouds
and phase profiles — without re-running ais++.

Supports 4D transverse phase-space grids (x0, y0, vx0, vy0) with z0/vz0
fixed, as produced by ais++ psgrid4d mode.

GPU acceleration via CuPy
--------------------------
When CuPy is available the **entire** ``generate_atoms`` pipeline runs on
the GPU:

  1. (CPU) Sample (x0, y0, vx0, vy0) with numpy.
  2. (GPU) Quadrilinear interpolation on the regular 4D PSMAP grid.
  3. (GPU) Port probabilities via _port_prob (cos, multiply, reduce).
  4. (GPU) Bernoulli sampling with CuPy's PRNG.
  5. (CPU) Thin host←device copy for the output DataFrame.

Typical usage
-------------
>>> from aispy.psmap import load_psmap, PSMAPSurrogate
>>> sur = PSMAPSurrogate(load_psmap('PSGRID4D_Z0.h5'), t_det=3.8)
>>> df  = sur.generate_atoms(mu_x0=0., mu_y0=0., mu_vx0=0., mu_vy0=0.,
...                          sigma_x=1e-4, sigma_y=1e-4,
...                          sigma_vx=3.1e-4, sigma_vy=3.1e-4,
...                          phi0=0., natoms=500_000)
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

    Works transparently with both numpy and cupy arrays.
    """
    xp = _get_xp(amp0)
    return (amp0**2 + amp1**2
            + is_interfering * 2.0 * amp0 * amp1 * xp.cos(dphi + delta))


# ── GPU interpolation on a regular grid ──────────────────────────────────────

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


def _catmull_weights(t, xp=np):
    """
    Catmull-Rom cubic spline weights for stencil offsets [-1, 0, 1, 2].
    Returns a list of four arrays [w_m1, w_0, w_1, w_2] each shaped like *t*.
    """
    t2 = t * t; t3 = t2 * t
    return [
        -0.5*t  +      t2 - 0.5*t3,
         1.0    - 2.5 *t2 + 1.5*t3,
         0.5*t  + 2.0 *t2 - 1.5*t3,
                - 0.5 *t2 + 0.5*t3,
    ]


def _quadrilinear(grid, ix, iy, ivx, ivy, tx, ty, tvx, tvy):
    """
    Quadrilinear interpolation on a 4D grid (x, y, vx, vy) using
    pre-computed cell indices and fractional offsets.  16-corner sum.
    All arrays may be cupy or numpy.
    """
    result = None
    for bx in range(2):
        wx = tx if bx else (1.0 - tx)
        for by in range(2):
            wy = ty if by else (1.0 - ty)
            for bvx in range(2):
                wvx = tvx if bvx else (1.0 - tvx)
                for bvy in range(2):
                    wvy = tvy if bvy else (1.0 - tvy)
                    corner = grid[ix+bx, iy+by, ivx+bvx, ivy+bvy]
                    term   = wx * wy * wvx * wvy * corner
                    result = term if result is None else result + term
    return result


def _quarticubic(grid, ix, iy, ivx, ivy, tx, ty, tvx, tvy, nx, ny, nvx, nvy):
    """
    Catmull-Rom cubic interpolation on a 4D grid (x, y, vx, vy).
    Uses a 4^4 = 256-corner stencil with clamped boundary conditions.
    All arrays may be cupy or numpy.
    """
    xp  = _get_xp(grid)
    wx  = _catmull_weights(tx,  xp)
    wy  = _catmull_weights(ty,  xp)
    wvx = _catmull_weights(tvx, xp)
    wvy = _catmull_weights(tvy, xp)
    result = None
    for bx in range(4):
        jx = xp.clip(ix + bx - 1, 0, nx - 1)
        for by in range(4):
            jy = xp.clip(iy + by - 1, 0, ny - 1)
            for bvx in range(4):
                jvx = xp.clip(ivx + bvx - 1, 0, nvx - 1)
                for bvy in range(4):
                    jvy = xp.clip(ivy + bvy - 1, 0, nvy - 1)
                    corner = grid[jx, jy, jvx, jvy]
                    w      = wx[bx] * wy[by] * wvx[bvx] * wvy[bvy]
                    result = w * corner if result is None else result + w * corner
    return result


# ── Surrogate ─────────────────────────────────────────────────────────────────

class PSMAPSurrogate:
    """
    Surrogate model built from an ais++ 4D transverse phase space map.

    The PSMAP stores, for every node (x0, y0, vx0, vy0) on a regular grid
    and every output port, the wavepacket amplitudes and the phase difference
    Δφ (computed in quad precision inside ais++).  z0 and vz0 are fixed.

    GPU acceleration
    ----------------
    When CuPy is available and ``use_gpu=True`` (the default), the residual
    grids are uploaded to the GPU at construction time and the **entire**
    ``generate_atoms`` pipeline runs on the GPU.

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

        # One row per atom (first occurrence of each atom index)
        first = np.searchsorted(atom_idx, np.arange(n_atoms))

        # Extract 4 transverse coordinates for each grid atom
        x0_atoms  = psmap['initial_positions'][first, 0]
        y0_atoms  = psmap['initial_positions'][first, 1]
        vx0_atoms = psmap['initial_velocities'][first, 0]
        vy0_atoms = psmap['initial_velocities'][first, 1]

        self.xs   = np.unique(x0_atoms)
        self.ys   = np.unique(y0_atoms)
        self.vxs  = np.unique(vx0_atoms)
        self.vys  = np.unique(vy0_atoms)
        self.nx   = len(self.xs)
        self.ny   = len(self.ys)
        self.nvx  = len(self.vxs)
        self.nvy  = len(self.vys)

        # Grid spacings (uniform by construction from ais++ linspace)
        self._x_lo  = float(self.xs[0])
        self._y_lo  = float(self.ys[0])
        self._vx_lo = float(self.vxs[0])
        self._vy_lo = float(self.vys[0])
        self._dx  = float((self.xs[-1]  - self.xs[0])  / (self.nx  - 1)) if self.nx  > 1 else 1.0
        self._dy  = float((self.ys[-1]  - self.ys[0])  / (self.ny  - 1)) if self.ny  > 1 else 1.0
        self._dvx = float((self.vxs[-1] - self.vxs[0]) / (self.nvx - 1)) if self.nvx > 1 else 1.0
        self._dvy = float((self.vys[-1] - self.vys[0]) / (self.nvy - 1)) if self.nvy > 1 else 1.0

        # Map each atom to its (ix, iy, ivx, ivy) grid index.
        # Using searchsorted is robust to any axis ordering in the ais++ output.
        xi  = np.searchsorted(self.xs,  x0_atoms)
        yi  = np.searchsorted(self.ys,  y0_atoms)
        vxi = np.searchsorted(self.vxs, vx0_atoms)
        vyi = np.searchsorted(self.vys, vy0_atoms)

        def to_grid(a):
            flat = a.reshape(n_atoms, nP)
            g = np.empty((self.nx, self.ny, self.nvx, self.nvy, nP))
            g[xi, yi, vxi, vyi, :] = flat
            return g

        dphi  = to_grid(psmap['phase_shifts'])
        amp0  = to_grid(psmap['amp0'])
        amp1  = to_grid(psmap['amp1'])
        inter = to_grid(psmap['is_interfering'].astype(float))
        state = to_grid(psmap['states'])

        self.port_states      = state[0, 0, 0, 0, :]
        self.port_interfering = inter[0, 0, 0, 0, :]
        self.port_path0 = psmap['path0'][first[0]:first[0]+nP]
        self.port_path1 = psmap['path1'][first[0]:first[0]+nP]

        shape = (self.nx, self.ny, self.nvx, self.nvy)
        X0g  = self.xs [:, None, None, None] * np.ones(shape)
        Y0g  = self.ys [None, :, None, None] * np.ones(shape)
        VX0g = self.vxs[None, None, :, None] * np.ones(shape)
        VY0g = self.vys[None, None, None, :] * np.ones(shape)

        # CPU interpolators (scipy handles N-D natively)
        axes = (self.xs, self.ys, self.vxs, self.vys)
        self._interp_dphi = [
            RegularGridInterpolator(axes, dphi[:,:,:,:,pi],
                                    method='cubic', bounds_error=False, fill_value=None)
            for pi in range(nP)]
        # Amplitudes are zeroed outside the grid (no extrapolation).
        self._interp_amp0 = [
            RegularGridInterpolator(axes, amp0[:,:,:,:,pi],
                                    method='cubic', bounds_error=False, fill_value=0.0)
            for pi in range(nP)]
        self._interp_amp1 = [
            RegularGridInterpolator(axes, amp1[:,:,:,:,pi],
                                    method='cubic', bounds_error=False, fill_value=0.0)
            for pi in range(nP)]

        # GPU: upload 4D grids once at construction time
        if self._use_gpu:
            self._gpu_dphi = [
                cp.asarray(dphi[:,:,:,:,pi], dtype=cp.float64)
                for pi in range(nP)]
            self._gpu_amp0 = [
                cp.asarray(amp0[:,:,:,:,pi], dtype=cp.float64)
                for pi in range(nP)]
            self._gpu_amp1 = [
                cp.asarray(amp1[:,:,:,:,pi], dtype=cp.float64)
                for pi in range(nP)]
            self._port_states_gpu = cp.asarray(self.port_states)
            self._port_inter_gpu  = cp.asarray(self.port_interfering)

        self._fit_metrics = self._compute_fit_metrics(dphi, X0g, Y0g, VX0g, VY0g)

    # ── Fit quality ──────────────────────────────────────────────────────────

    def _compute_fit_metrics(self, dphi, X0g, Y0g, VX0g, VY0g):
        x0f  = X0g.ravel();  y0f  = Y0g.ravel()
        vx0f = VX0g.ravel(); vy0f = VY0g.ravel()
        # Full 2nd-order polynomial in 4 variables (15 terms)
        A = np.column_stack([
            np.ones_like(x0f),
            x0f, y0f, vx0f, vy0f,
            x0f**2, x0f*y0f, x0f*vx0f, x0f*vy0f,
            y0f**2, y0f*vx0f, y0f*vy0f,
            vx0f**2, vx0f*vy0f, vy0f**2,
        ])
        n_atoms = len(x0f)
        metrics = {}
        for pi in range(self.nP):
            d      = dphi[:, :, :, :, pi].ravel()
            c, *_  = np.linalg.lstsq(A, d, rcond=None)
            pred   = A @ c
            ss_res = np.sum((d - pred)**2)
            ss_tot = np.sum((d - d.mean())**2)
            metrics[pi] = {
                'r2':               float(1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0),
                'residual_std_rad': float(np.sqrt(ss_res / n_atoms)),
                'quad_coeffs':      c,
            }
        return metrics

    def fit_quality(self):
        """
        Return quality metrics for a quadratic fit to Δφ(x0, y0, vx0, vy0).

        Returns
        -------
        dict keyed by port index with sub-keys ``r2`` and
        ``residual_std_rad``.
        """
        return {pi: {k: v for k, v in m.items() if k != 'quad_coeffs'}
                for pi, m in self._fit_metrics.items()}

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _eval_gpu(self, x0_g, y0_g, vx0_g, vy0_g):
        """Catmull-Rom cubic interpolation on GPU."""
        ix,  tx  = _find_cell(x0_g,  self._x_lo,  self._dx,  self.nx)
        iy,  ty  = _find_cell(y0_g,  self._y_lo,  self._dy,  self.ny)
        ivx, tvx = _find_cell(vx0_g, self._vx_lo, self._dvx, self.nvx)
        ivy, tvy = _find_cell(vy0_g, self._vy_lo, self._dvy, self.nvy)

        N = len(x0_g)
        dphi_out = cp.empty((N, self.nP), dtype=cp.float64)
        amp0_out = cp.empty((N, self.nP), dtype=cp.float64)
        amp1_out = cp.empty((N, self.nP), dtype=cp.float64)

        dims = (self.nx, self.ny, self.nvx, self.nvy)
        for pi in range(self.nP):
            dphi_out[:, pi] = _quarticubic(
                self._gpu_dphi[pi], ix, iy, ivx, ivy, tx, ty, tvx, tvy, *dims)
            amp0_out[:, pi] = _quarticubic(
                self._gpu_amp0[pi], ix, iy, ivx, ivy, tx, ty, tvx, tvy, *dims)
            amp1_out[:, pi] = _quarticubic(
                self._gpu_amp1[pi], ix, iy, ivx, ivy, tx, ty, tvx, tvy, *dims)

        # Zero amplitudes outside the grid — dphi extrapolation is irrelevant there.
        out_mask = (
            (x0_g  < self.xs[0])  | (x0_g  > self.xs[-1])  |
            (y0_g  < self.ys[0])  | (y0_g  > self.ys[-1])  |
            (vx0_g < self.vxs[0]) | (vx0_g > self.vxs[-1]) |
            (vy0_g < self.vys[0]) | (vy0_g > self.vys[-1])
        )
        amp0_out[out_mask] = 0.0
        amp1_out[out_mask] = 0.0

        return dphi_out, amp0_out, amp1_out

    def eval(self, x0_arr, y0_arr, vx0_arr, vy0_arr):
        """
        Evaluate the surrogate at arbitrary (x0, y0, vx0, vy0) coordinates.

        Uses GPU quadrilinear interpolation when ``use_gpu=True``, scipy on
        CPU otherwise.  Always returns **numpy** arrays.

        Parameters
        ----------
        x0_arr, y0_arr, vx0_arr, vy0_arr : array-like, shape (N,)

        Returns
        -------
        dphi  : ndarray (N, nP)
        amp0  : ndarray (N, nP)
        amp1  : ndarray (N, nP)
        """
        x0_arr  = np.asarray(x0_arr,  dtype=float)
        y0_arr  = np.asarray(y0_arr,  dtype=float)
        vx0_arr = np.asarray(vx0_arr, dtype=float)
        vy0_arr = np.asarray(vy0_arr, dtype=float)

        if self._use_gpu:
            dphi_g, amp0_g, amp1_g = self._eval_gpu(
                cp.asarray(x0_arr),  cp.asarray(y0_arr),
                cp.asarray(vx0_arr), cp.asarray(vy0_arr))
            return cp.asnumpy(dphi_g), cp.asnumpy(amp0_g), cp.asnumpy(amp1_g)

        pts = np.column_stack([x0_arr, y0_arr, vx0_arr, vy0_arr])
        N   = len(x0_arr)
        dphi_out = np.empty((N, self.nP))
        amp0_out = np.empty((N, self.nP))
        amp1_out = np.empty((N, self.nP))
        for pi in range(self.nP):
            dphi_out[:, pi] = self._interp_dphi[pi](pts)
            amp0_out[:, pi] = self._interp_amp0[pi](pts)
            amp1_out[:, pi] = self._interp_amp1[pi](pts)
        return dphi_out, amp0_out, amp1_out

    # ── Atom generation ───────────────────────────────────────────────────────

    def generate_atoms(self, mu_x0, mu_y0, mu_vx0, mu_vy0,
                       sigma_x, sigma_y, sigma_vx, sigma_vy,
                       phi0=0.0, natoms=100_000,
                       phase_profile=None, rng=None,
                       _return_arrays=False, _image_edges=None):
        """
        Sample atoms from a Gaussian cloud and return synthetic observations.

        Because ais++ prunes all but the two main interferometry paths
        (usepathselection 1), p0 + p1 ≤ 1 in general — the deficit
        1 - p0 - p1 is probability that ended up in pruned paths, i.e.
        atoms that would not be detected.  The correct sampling is therefore
        a three-outcome draw per atom:
          - detected in ground state  with prob p0
          - detected in excited state with prob p1
          - not detected (lost)       with prob 1 - p0 - p1

        Only detected atoms are returned, so the output DataFrame has fewer
        than ``natoms`` rows in general.  ``prob_det`` gives the detection
        probability for each returned atom (useful for importance weighting);
        ``prob_s0`` is the conditional ground-state probability p0/(p0+p1).

        GPU image path (use_gpu=True and _image_edges is not None)  ← fastest
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1. GPU  — sample (x0, y0, vx0, vy0); free-flight to (xf, yf).
        2. GPU  — quadrilinear interpolation of dphi, amp0, amp1.
        3. GPU  — port probabilities; detection + state Bernoulli draws.
        4. GPU  — 2D histogram into (res × res) bins using _image_edges.
        5. CPU  — device→host for two (res, res) uint16 images only.

        At 10^8 atoms the detected arrays are ~1.6 GB; the two images are
        ~8 MB.  Histogramming on the GPU before D2H reduces transfer and
        CPU work from ~14 s to ~1 s.

        GPU array path (use_gpu=True and _return_arrays=True)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1. GPU  — sample (x0, y0, vx0, vy0); free-flight to (xf, yf).
        2. GPU  — quadrilinear interpolation of dphi, amp0, amp1.
        3. GPU  — port probabilities; detection + state Bernoulli draws.
        4. CPU  — device→host for detected atoms only (state, xf, yf).

        GPU DataFrame path (use_gpu=True, _return_arrays=False, no _image_edges)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1. CPU  — sample (x0, y0, vx0, vy0); free-flight to (xf, yf).
        2. CPU  — evaluate phase_profile(xf, yf, vxf, vyf) if provided.
        3. GPU  — host→device transfer.
        4. GPU  — quadrilinear interpolation of dphi, amp0, amp1.
        5. GPU  — port probabilities; detection + state Bernoulli draws.
        6. CPU  — device→host; filter to detected atoms only.

        Parameters
        ----------
        mu_x0, mu_y0         : float  Cloud COM [m]
        mu_vx0, mu_vy0       : float  Cloud COM velocity [m/s]
        sigma_x, sigma_y     : float  Cloud 1σ position spreads [m]
        sigma_vx, sigma_vy   : float  Cloud 1σ velocity spreads [m/s]
        phi0                 : float  Global phase offset [rad]
        natoms               : int    Number of atoms launched (detected count
                                      will be lower)
        phase_profile        : callable or None
            ``f(xf, yf, vxf, vyf) -> array (natoms,)`` [rad].
        rng                  : numpy Generator, int seed, or None
        _return_arrays       : bool
            When True and use_gpu=True, return (states, xf, yf) as numpy
            arrays instead of a DataFrame (GPU array path).
            Ignored when _image_edges is provided.
        _image_edges         : array-like of shape (res+1,), or None
            Symmetric bin edges for both x and y axes [m].  When provided,
            the 2D histogram is computed on the GPU (or CPU when use_gpu is
            False) and the function returns (img_s0, img_s1) uint16 images
            instead of per-atom data.  Takes precedence over _return_arrays.

        Returns
        -------
        If _image_edges is not None:
            tuple (img_s0, img_s1) — numpy uint16 arrays of shape (res, res).
                img_s0  ground-state (state=0) atom counts per pixel
                img_s1  excited-state (state=1) atom counts per pixel
            Atoms outside the bin range are silently dropped (same as
            np.histogram2d behaviour).
        If _return_arrays=True and use_gpu=True:
            tuple (states, xf, yf) — numpy arrays for detected atoms only.
                states : int8  (0=ground, 1=excited)
                xf     : float64  final x position [m]
                yf     : float64  final y position [m]
        Otherwise:
            pandas.DataFrame (detected atoms only) with columns:
                x0, y0, vx0, vy0   Initial phase-space coordinates [m, m/s]
                xf, yf, vxf, vyf   Final transverse coordinates [m, m/s]
                state               Detected output state (0=ground, 1=excited)
                prob_s0             Conditional P(ground | detected) = p0/(p0+p1)
                prob_det            Detection probability p0 + p1 for this atom
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # ── GPU fast path: sample + interpolate entirely on device ──────────
        if self._use_gpu and (_return_arrays or _image_edges is not None):
            x0_g  = cp.random.normal(mu_x0,  sigma_x,  natoms, dtype=cp.float64)
            y0_g  = cp.random.normal(mu_y0,  sigma_y,  natoms, dtype=cp.float64)
            vx0_g = cp.random.normal(mu_vx0, sigma_vx, natoms, dtype=cp.float64)
            vy0_g = cp.random.normal(mu_vy0, sigma_vy, natoms, dtype=cp.float64)
            xf_g  = x0_g + vx0_g * self.t_det
            yf_g  = y0_g + vy0_g * self.t_det

            if phase_profile is not None:
                delta_g = cp.asarray(np.asarray(
                    phase_profile(cp.asnumpy(xf_g), cp.asnumpy(yf_g),
                                  cp.asnumpy(vx0_g), cp.asnumpy(vy0_g)),
                    dtype=np.float64))
            else:
                delta_g = cp.zeros(natoms, dtype=cp.float64)

            dphi_g, amp0_g, amp1_g = self._eval_gpu(x0_g, y0_g, vx0_g, vy0_g)
            total_phase = dphi_g + (delta_g + phi0)[:, None]
            probs = _port_prob(amp0_g, amp1_g, dphi_g,
                               self._port_inter_gpu[None, :],
                               delta=total_phase - dphi_g)
            s0_mask    = (self._port_states_gpu == 0)
            p0         = cp.clip(probs[:, s0_mask ].sum(axis=1), 0.0, None)
            p1         = cp.clip(probs[:, ~s0_mask].sum(axis=1), 0.0, None)
            prob_det_g = cp.clip(p0 + p1, 0.0, 1.0)

            u1         = cp.random.random(natoms, dtype=cp.float64)
            u2         = cp.random.random(natoms, dtype=cp.float64)
            det_mask_g = u1 < prob_det_g
            prob_s0_g  = cp.where(prob_det_g > 0,
                                  p0 / cp.maximum(prob_det_g, 1e-300), 0.5)
            state_g    = (u2 > prob_s0_g).astype(cp.int8)

            if _image_edges is not None:
                # Histogram on GPU before D2H: transfers ~8 MB instead of ~1.6 GB
                # at 10^8 atoms, saving ~14 s of CPU histogram work.
                edges_g  = cp.asarray(_image_edges, dtype=cp.float64)
                xf_det   = xf_g[det_mask_g]
                yf_det   = yf_g[det_mask_g]
                s0_det_g = (state_g[det_mask_g] == 0)
                h0, _, _ = cp.histogram2d(xf_det[ s0_det_g], yf_det[ s0_det_g], bins=edges_g)
                h1, _, _ = cp.histogram2d(xf_det[~s0_det_g], yf_det[~s0_det_g], bins=edges_g)
                return h0.get().astype(np.uint16), h1.get().astype(np.uint16)

            return (
                cp.asnumpy(state_g[det_mask_g]),
                cp.asnumpy(xf_g[det_mask_g]),
                cp.asnumpy(yf_g[det_mask_g]),
            )

        # ── Standard path: CPU sampling ───────────────────────────────────────
        import pandas as pd

        # ── 1. Sample + free-flight kinematic map (CPU) ───────────────────────
        x0  = rng.normal(mu_x0,  sigma_x,  natoms)
        y0  = rng.normal(mu_y0,  sigma_y,  natoms)
        vx0 = rng.normal(mu_vx0, sigma_vx, natoms)
        vy0 = rng.normal(mu_vy0, sigma_vy, natoms)
        xf  = x0  + vx0 * self.t_det
        yf  = y0  + vy0 * self.t_det
        vxf = vx0.copy()
        vyf = vy0.copy()

        # ── 2. Phase profile (CPU — user callable receives numpy) ─────────────
        delta_np = (np.asarray(phase_profile(xf, yf, vxf, vyf), dtype=np.float64)
                    if phase_profile is not None
                    else np.zeros(natoms, dtype=np.float64))

        if self._use_gpu:
            # ── 3. Host → device ─────────────────────────────────────────────
            x0_g  = cp.asarray(x0,       dtype=cp.float64)
            y0_g  = cp.asarray(y0,       dtype=cp.float64)
            vx0_g = cp.asarray(vx0,      dtype=cp.float64)
            vy0_g = cp.asarray(vy0,      dtype=cp.float64)
            delta_g = cp.asarray(delta_np, dtype=cp.float64)

            # ── 4. GPU quadrilinear interpolation ─────────────────────────────
            dphi_g, amp0_g, amp1_g = self._eval_gpu(x0_g, y0_g, vx0_g, vy0_g)

            # ── 5. Port probabilities ─────────────────────────────────────────
            total_phase = dphi_g + (delta_g + phi0)[:, None]
            probs = _port_prob(amp0_g, amp1_g, dphi_g,
                               self._port_inter_gpu[None, :],
                               delta=total_phase - dphi_g)
            s0_mask = (self._port_states_gpu == 0)
            p0 = cp.clip(probs[:, s0_mask ].sum(axis=1), 0.0, None)
            p1 = cp.clip(probs[:, ~s0_mask].sum(axis=1), 0.0, None)
            prob_det_g = cp.clip(p0 + p1, 0.0, 1.0)

            u1 = cp.random.random(natoms, dtype=cp.float64)
            u2 = cp.random.random(natoms, dtype=cp.float64)
            det_mask_g = u1 < prob_det_g
            prob_s0_cond_g = cp.where(prob_det_g > 0,
                                      p0 / cp.maximum(prob_det_g, 1e-300), 0.5)
            state_g = (u2 > prob_s0_cond_g).astype(cp.int8)

            # ── 6. Device → host; keep only detected atoms ────────────────────
            det_mask_np    = cp.asnumpy(det_mask_g)
            state_np       = cp.asnumpy(state_g)[det_mask_np]
            prob_s0_np     = cp.asnumpy(prob_s0_cond_g)[det_mask_np]
            prob_det_np    = cp.asnumpy(prob_det_g)[det_mask_np]

        else:
            pts = np.column_stack([x0, y0, vx0, vy0])
            dphi_np = np.empty((natoms, self.nP))
            amp0_np = np.empty((natoms, self.nP))
            amp1_np = np.empty((natoms, self.nP))
            for pi in range(self.nP):
                dphi_np[:, pi] = self._interp_dphi[pi](pts)
                amp0_np[:, pi] = self._interp_amp0[pi](pts)
                amp1_np[:, pi] = self._interp_amp1[pi](pts)

            total_phase = dphi_np + (delta_np + phi0)[:, None]
            probs = _port_prob(amp0_np, amp1_np, dphi_np,
                               self.port_interfering[None, :],
                               delta=total_phase - dphi_np)
            s0_mask  = (self.port_states == 0)
            p0 = np.clip(probs[:, s0_mask ].sum(axis=1), 0.0, None)
            p1 = np.clip(probs[:, ~s0_mask].sum(axis=1), 0.0, None)
            prob_det_np = np.clip(p0 + p1, 0.0, 1.0)

            u1 = rng.uniform(size=natoms)
            u2 = rng.uniform(size=natoms)
            det_mask_np = u1 < prob_det_np
            prob_s0_np  = np.where(prob_det_np > 0,
                                   p0 / np.maximum(prob_det_np, 1e-300), 0.5)
            state_np    = (u2 > prob_s0_np).astype(np.int8)

            state_np    = state_np[det_mask_np]
            prob_s0_np  = prob_s0_np[det_mask_np]
            prob_det_np = prob_det_np[det_mask_np]

        if _image_edges is not None:
            edges    = np.asarray(_image_edges)
            xf_det   = xf[det_mask_np]
            yf_det   = yf[det_mask_np]
            s0       = (state_np == 0)
            h0, _, _ = np.histogram2d(xf_det[ s0], yf_det[ s0], bins=edges)
            h1, _, _ = np.histogram2d(xf_det[~s0], yf_det[~s0], bins=edges)
            return h0.astype(np.uint16), h1.astype(np.uint16)

        return pd.DataFrame({
            'x0':  x0[det_mask_np],  'y0':  y0[det_mask_np],
            'vx0': vx0[det_mask_np], 'vy0': vy0[det_mask_np],
            'xf':  xf[det_mask_np],  'yf':  yf[det_mask_np],
            'vxf': vxf[det_mask_np], 'vyf': vyf[det_mask_np],
            'state':    state_np,
            'prob_s0':  prob_s0_np,
            'prob_det': prob_det_np,
        })
