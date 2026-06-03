"""
Phase space map surrogate for ais++ simulations.

Loads a _PSMAP.h5 file produced by ais++ in ``psgrid`` mode, builds
per-port interpolators for the wavepacket amplitudes and phase shifts,
and generates synthetic atom observations for arbitrary Gaussian clouds
and phase profiles — without re-running ais++.

Typical usage
-------------
>>> from aispy.psmap import load_psmap, PSMAPSurrogate
>>> data = load_psmap('PSR_EXAMPLE_PSGRID.h5')
>>> sur  = PSMAPSurrogate(data, t_det=4.451)
>>> df   = sur.generate_atoms(mu_x0=0., mu_vx0=0.,
...                           sigma_x=1e-4, sigma_vx=3.1e-4,
...                           phi0=0., natoms=200_000,
...                           phase_profile=lambda xf, vxf: 3140. * xf)
"""

import warnings
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _port_prob(amp0, amp1, dphi, is_interfering, delta=0.0):
    """
    Port probability P = A0² + A1² + interfering·2A0A1·cos(Δφ + δ).

    All arguments broadcast normally.
    """
    return (amp0**2 + amp1**2
            + is_interfering * 2.0 * amp0 * amp1 * np.cos(dphi + delta))


def _linspace_grid(arr):
    """Return the unique sorted values of a 1-D flat array."""
    return np.unique(arr)


# ── Surrogate ─────────────────────────────────────────────────────────────────

class PSMAPSurrogate:
    """
    Surrogate model built from an ais++ phase space map.

    The PSMAP stores, for every node (x0, vx0) on a regular grid and every
    output port, the wavepacket amplitudes and the phase difference Δφ
    (computed in quad precision inside ais++).

    This class builds a ``RegularGridInterpolator`` for each quantity so
    that the surrogate can be evaluated at arbitrary (x0, vx0) — i.e. for
    atoms drawn from a continuous Gaussian cloud.

    Parameters
    ----------
    psmap : dict
        Output of :func:`load_psmap`.
    t_det : float
        Detection time [s].  Used for the exact kinematic mapping
        ``xf = x0 + vx0 * t_det``, ``vxf = vx0``.
    """

    def __init__(self, psmap, t_det):
        self.t_det = float(t_det)

        atom_idx = psmap['atom_indices']
        n_atoms  = int(atom_idx.max()) + 1
        nP       = int(np.bincount(atom_idx)[0])
        self.nP  = nP

        # Identify regular (x0, vx0) grid
        first     = np.searchsorted(atom_idx, np.arange(n_atoms))
        x0_atoms  = psmap['initial_positions'][first, 0]
        vx0_atoms = psmap['initial_velocities'][first, 0]
        self.xs   = _linspace_grid(x0_atoms)
        self.vxs  = _linspace_grid(vx0_atoms)
        self.nx   = len(self.xs)
        self.nvx  = len(self.vxs)

        def to_grid(a):
            return a.reshape(n_atoms, nP).reshape(self.nx, self.nvx, nP)

        dphi  = to_grid(psmap['phase_shifts'])
        amp0  = to_grid(psmap['amp0'])
        amp1  = to_grid(psmap['amp1'])
        inter = to_grid(psmap['is_interfering'].astype(float))
        state = to_grid(psmap['states'])

        # Port metadata — identical for all atoms in grid mode
        self.port_states      = state[0, 0, :]          # (nP,) int
        self.port_interfering = inter[0, 0, :]          # (nP,) float {0,1}
        self.port_path0 = psmap['path0'][first[0]:first[0]+nP]
        self.port_path1 = psmap['path1'][first[0]:first[0]+nP]

        # Grid coordinates for vectorised linear-term removal
        X0g  = self.xs[:, None]  * np.ones((1, self.nvx))
        VX0g = np.ones((self.nx, 1)) * self.vxs[None, :]

        # Per-port interpolators
        self._dphi_linear = np.zeros((nP, 3))   # [a0, a1·x0, a2·vx0]
        self._interp_dphi = []
        self._interp_amp0 = []
        self._interp_amp1 = []

        for pi in range(nP):
            d = dphi[:, :, pi]

            # Strip dominant linear trend before interpolating so the
            # interpolator sees a smooth O(few mrad) residual rather
            # than a ~4e8 rad background.
            A  = np.column_stack([np.ones(self.nx * self.nvx),
                                  X0g.ravel(), VX0g.ravel()])
            c, *_ = np.linalg.lstsq(A, d.ravel(), rcond=None)
            self._dphi_linear[pi] = c
            resid = d - (c[0] + c[1]*X0g + c[2]*VX0g)

            kw = dict(method='linear', bounds_error=False, fill_value=None)
            self._interp_dphi.append(
                RegularGridInterpolator((self.xs, self.vxs), resid,  **kw))
            self._interp_amp0.append(
                RegularGridInterpolator((self.xs, self.vxs), amp0[:,:,pi], **kw))
            self._interp_amp1.append(
                RegularGridInterpolator((self.xs, self.vxs), amp1[:,:,pi], **kw))

        # Quadratic fit quality (precomputed for all ports)
        self._fit_metrics = self._compute_fit_metrics(dphi, X0g, VX0g)

    # ── Fit quality ──────────────────────────────────────────────────────────

    def _compute_fit_metrics(self, dphi, X0g, VX0g):
        x0f  = X0g.ravel()
        vx0f = VX0g.ravel()
        A = np.column_stack([np.ones_like(x0f),
                             x0f, vx0f,
                             x0f**2, x0f*vx0f, vx0f**2])
        metrics = {}
        for pi in range(self.nP):
            d = dphi[:, :, pi].ravel()
            c, *_ = np.linalg.lstsq(A, d, rcond=None)
            pred       = A @ c
            ss_res     = np.sum((d - pred)**2)
            ss_tot     = np.sum((d - d.mean())**2)
            r2         = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            metrics[pi] = {
                'r2':               float(r2),
                'residual_std_rad': float(np.sqrt(ss_res / len(d))),
                'quad_coeffs':      c,
            }
        return metrics

    def fit_quality(self):
        """
        Return quality metrics for a quadratic fit to Δφ(x0, vx0).

        A high R² (close to 1) means the grid data is well described by
        the analytical Gaussian-beam model from Eq. 11 of Mouelle et al.
        The residual std gives the mismatch in rad.

        Returns
        -------
        dict keyed by port index with sub-keys ``r2`` and
        ``residual_std_rad``.
        """
        return {pi: {k: v for k, v in m.items() if k != 'quad_coeffs'}
                for pi, m in self._fit_metrics.items()}

    # ── Evaluation ───────────────────────────────────────────────────────────

    def eval(self, x0_arr, vx0_arr):
        """
        Evaluate the surrogate at arbitrary (x0, vx0) coordinates.

        Atoms outside the grid are extrapolated from the boundary (nearest
        edge value of each interpolator).  A warning is raised if more than
        1 % of points lie outside.

        Parameters
        ----------
        x0_arr, vx0_arr : array-like, shape (N,)

        Returns
        -------
        dphi  : ndarray (N, nP)
        amp0  : ndarray (N, nP)
        amp1  : ndarray (N, nP)
        """
        x0_arr  = np.asarray(x0_arr, dtype=float)
        vx0_arr = np.asarray(vx0_arr, dtype=float)

        out_frac = (
            np.mean((x0_arr  < self.xs[0])  | (x0_arr  > self.xs[-1]))
          + np.mean((vx0_arr < self.vxs[0]) | (vx0_arr > self.vxs[-1]))
        ) / 2.0
        if out_frac > 0.01:
            warnings.warn(
                f'{out_frac*100:.1f}% of sampled atoms lie outside the '
                'PSMAP grid and will be extrapolated. Consider extending '
                'the grid limits.', stacklevel=2)

        pts = np.column_stack([x0_arr, vx0_arr])
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

        For each atom the transverse position at detection is computed
        analytically as ``xf = x0 + vx0 * t_det`` (exact for free
        transverse motion).  The phase profile, if given, is evaluated at
        ``(xf, vxf)`` and added to Δφ before computing the port probability.

        Parameters
        ----------
        mu_x0, mu_vx0    : float   Cloud centre-of-mass [m, m/s]
        sigma_x, sigma_vx : float  Cloud 1σ spreads [m, m/s]
        phi0              : float  Global interferometer phase offset [rad]
        natoms            : int    Number of atoms to simulate
        phase_profile     : callable or None
            ``f(xf, vxf) -> ndarray (natoms,)`` in radians.
            Applied on top of Δφ + φ0.
            Example — phase shear: ``lambda xf, vxf: kappa * xf``
        rng               : numpy Generator, int seed, or None

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

        # ── 1. Sample initial conditions ─────────────────────────────────────
        x0  = rng.normal(mu_x0,  sigma_x,  natoms)
        vx0 = rng.normal(mu_vx0, sigma_vx, natoms)

        # ── 2. Analytical kinematic mapping ──────────────────────────────────
        xf  = x0 + vx0 * self.t_det
        vxf = vx0.copy()

        # ── 3. Interpolate PSMAP quantities ──────────────────────────────────
        dphi_s, amp0_s, amp1_s = self.eval(x0, vx0)   # (natoms, nP)

        # ── 4. Phase profile at final position ───────────────────────────────
        delta = np.zeros(natoms)
        if phase_profile is not None:
            delta = np.asarray(phase_profile(xf, vxf), dtype=float)

        # ── 5. Port probabilities ─────────────────────────────────────────────
        # delta and phi0 broadcast over ports (same value for all ports of
        # each atom — the profile is applied to the full complex amplitude,
        # not differentially between wavepackets)
        total_phase = dphi_s + (delta + phi0)[:, None]   # (natoms, nP)
        probs = _port_prob(amp0_s, amp1_s, dphi_s,
                           self.port_interfering[None, :],
                           delta=total_phase - dphi_s)    # (natoms, nP)

        # ── 6. Ground-state probability ───────────────────────────────────────
        s0_mask = (self.port_states == 0)
        prob_s0 = np.clip(probs[:, s0_mask].sum(axis=1), 0.0, 1.0)

        # ── 7. Bernoulli sample ───────────────────────────────────────────────
        state = (rng.uniform(size=natoms) > prob_s0).astype(np.int8)

        return pd.DataFrame({'x0': x0, 'vx0': vx0,
                             'xf': xf, 'vxf': vxf,
                             'state': state, 'prob_s0': prob_s0})
