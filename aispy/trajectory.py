"""
Interferometer trajectory loading and plotting.

Loads _TRAJ.h5 files produced by ais++ with ``printtrajectory 1`` and
reconstructs smooth spacetime trajectories by analytically evaluating the
free-flight motion between pulse-boundary snapshots.

For ``linear_pot`` (uniform gravity) the free flight is exact:
  x(t) = x0 + vx * dt
  z(t) = z0 + vz * dt - ½ g dt²

so knowing (pos, vel) at each pulse boundary is sufficient to draw a
perfectly smooth trajectory at any resolution without approximation.

For the ``rotating_*`` potentials the equations of motion are still linear with
constant coefficients, so the free flight is again exact — see
:func:`_free_flight_factory`, which propagates with a matrix exponential rather
than a parabola. The frame rotation rate is read from the ``rotation`` dataset
that ais++ writes into every output file, so rotating runs need no extra
arguments.

Typical usage
-------------
>>> from aispy.trajectory import load_trajectory, plot_trajectory
>>> fig, axes = plot_trajectory('output-files/TRAJ_MZ_N1_TRAJ.h5')
>>> fig.savefig('trajectory.png', dpi=150, bbox_inches='tight')
"""

import numpy as np
import h5py


# ── constants ─────────────────────────────────────────────────────────────────
_G = 9.81   # m/s²


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_trajectory(fname):
    """
    Load a _TRAJ.h5 file.

    Parameters
    ----------
    fname : str or Path

    Returns
    -------
    dict with keys:
        snapshot_times  (S,)      float64 — time of each snapshot [s]
        snapshot_labels (S,)      list[str]
        snapshot_idx    (N,)      int — which snapshot each record belongs to
        atom_indices    (N,)      int
        paths           (N,)      list[str]
        states          (N,)      int
        amplitudes      (N,)      float64
        positions       (N, 3)    float64 [m]
        velocities      (N, 3)    float64 [m/s]  — frame velocities
        rotation        (3,)      float64 [rad/s] — frame angular velocity
                                  (zeros for files written before rotating-frame
                                  support, or for inertial runs)
    """
    with h5py.File(fname, 'r') as f:
        decode = lambda arr: [x.decode() if isinstance(x, bytes) else x
                              for x in arr]
        rotation = (f['rotation'][:] if 'rotation' in f
                    else np.zeros(3))
        return {
            'snapshot_times':  f['snapshot_times'][:],
            'snapshot_labels': decode(f['snapshot_labels'][:]),
            'snapshot_idx':    f['snapshot_idx'][:],
            'atom_indices':    f['atom_indices'][:],
            'paths':           decode(f['paths'][:]),
            'states':          f['states'][:],
            'amplitudes':      f['amplitudes'][:],
            'positions':       f['positions'][:],
            'velocities':      f['velocities'][:],
            'rotation':        np.asarray(rotation, dtype=float),
        }


# ── Free flight ───────────────────────────────────────────────────────────────

def _cross_matrix(w):
    """Skew matrix W with ``W @ a == np.cross(w, a)``."""
    wx, wy, wz = w
    return np.array([[0.0, -wz,  wy],
                     [ wz, 0.0, -wx],
                     [-wy,  wx, 0.0]])


def _free_flight_factory(g, rotation):
    """
    Build an exact free-flight propagator ``f(p0, v0, t0, t_arr)``.

    ``p0`` is a frame position and ``v0`` a frame velocity, matching what ais++
    writes into the snapshots. Returns ``(x, y, z)`` arrays at the times in
    *t_arr*.

    With no rotation this is the familiar parabola. With rotation the equations
    of motion in canonical variables (v = p/m = xdot + Omega x x) are

        d/dt x = v - Omega x x
        d/dt v = -Omega x v - g zhat

    which is linear with constant coefficients and a constant forcing term, so
    it is solved exactly by one matrix exponential of the augmented system

        d/dt [x; v; 1] = A [x; v; 1].

    That is exact, not an approximation, and reduces to the parabola as
    Omega -> 0.
    """
    rotation = np.asarray(rotation, dtype=float)

    if not np.any(rotation):
        def free_flight(p0, v0, t0, t_arr):
            dt = np.asarray(t_arr) - t0
            return (p0[0] + v0[0]*dt,
                    p0[1] + v0[1]*dt,
                    p0[2] + v0[2]*dt - 0.5*g*dt**2)
        return free_flight

    from scipy.linalg import expm

    W = _cross_matrix(rotation)
    A = np.zeros((7, 7))
    A[0:3, 0:3] = -W
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 3:6] = -W
    A[3:6, 6]   = [0.0, 0.0, -g]

    def free_flight(p0, v0, t0, t_arr):
        t_arr = np.asarray(t_arr, dtype=float)
        if t_arr.size == 0:
            return (t_arr.copy(), t_arr.copy(), t_arr.copy())

        p0 = np.asarray(p0, dtype=float)
        v0 = np.asarray(v0, dtype=float)
        # snapshots store frame velocities; the propagator needs canonical ones
        y = np.empty(7)
        y[0:3] = p0
        y[3:6] = v0 + np.cross(rotation, p0)
        y[6]   = 1.0

        dts = t_arr - t0
        uniform = (t_arr.size > 1 and
                   np.allclose(np.diff(dts), dts[1] - dts[0], rtol=1e-9, atol=0.0))

        out = np.empty((t_arr.size, 3))
        if uniform:
            # one expm, then repeated application along the uniform grid
            step = expm(A * (dts[1] - dts[0]))
            state = expm(A * dts[0]) @ y
            out[0] = state[0:3]
            for i in range(1, t_arr.size):
                state = step @ state
                out[i] = state[0:3]
        else:
            for i, dt in enumerate(dts):
                out[i] = (expm(A * dt) @ y)[0:3]

        return out[:, 0], out[:, 1], out[:, 2]

    return free_flight


# ── Reconstruction ────────────────────────────────────────────────────────────

def _arm_key(path):
    """
    Return a stable arm identifier from a (possibly growing) path string.
    'init'   — before the first beamsplitter (path length <= 1)
    'lower'  — path[1] == '0' (lower arm throughout LMT sequence)
    'upper'  — path[1] == '1' (upper arm throughout LMT sequence)
    """
    if len(path) <= 1:
        return 'init'
    return 'lower' if path[1] == '0' else 'upper'


def reconstruct_trajectories(traj, atom_idx=0, potential='linear_pot',
                              n_interp=200, min_interp_dt=0.005,
                              arm_grouping='auto', rotation=None):
    """
    Reconstruct smooth (t, x, y, z) curves from snapshot data.

    Grouping modes
    --------------
    ``arm_grouping='auto'`` (default)
        Use **arm-level grouping** (path[1]) when the longest path string
        exceeds 10 characters — i.e. for LMT sequences where the path grows
        at every pulse.  This reduces 8000+ distinct path strings to just 2
        arms (+ the initial wavepacket), giving O(n) matching and 2–3
        matplotlib plot() calls instead of 8000.

        Use **exact-path grouping** for short sequences (n=1 simple MZ),
        preserving the individual output-port lines.

    ``arm_grouping=True``  — always group by arm
    ``arm_grouping=False`` — always group by exact path string

    Arm grouping at the final beamsplitter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    After the last beamsplitter each arm produces 2 wavepackets (one per
    output state).  Both land at the same z position, so the arm trajectory
    is closed by averaging their positions at the final snapshot.  The
    ``state`` field in the result reflects the final state of the lower-
    amplitude wavepacket for that arm (typically meaningless for LMT plots;
    color by arm instead of state when using arm grouping).

    Free-flight interpolation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Analytic free-flight is only evaluated for intervals longer than
    *min_interp_dt* (default 5 ms).  Short LMT pulse gaps are left as-is
    (the dense snapshots already form a smooth curve); long free-flight
    intervals get *n_interp* parabolic fill-points.

    Parameters
    ----------
    traj           : dict   — output of :func:`load_trajectory`
    atom_idx       : int    — which atom to reconstruct (default 0)
    potential      : str    — ``'zero_pot'``/``'rotating_pot'`` (no gravity) or
                              ``'linear_pot'``/``'rotating_linear_pot'`` (uniform g)
    n_interp       : int    — interpolation points per free-flight segment
    min_interp_dt  : float  — minimum interval [s] to interpolate (default 5 ms)
    arm_grouping   : str or bool — ``'auto'``, ``True``, or ``False``
    rotation       : (3,) array or None — frame angular velocity [rad/s].
                     ``None`` (default) takes it from ``traj['rotation']``.

    Returns
    -------
    dict keyed by group label (path string or arm key), each value a dict:
        t          (M,) float64
        x, y, z    (M,) float64  [m]
        state       int
        amplitude   float64
        arm_grouped bool  — True when arm-level grouping was used
    """
    snap_times = traj['snapshot_times']
    mask = np.asarray(traj['atom_indices']) == atom_idx

    snap_idx  = np.asarray(traj['snapshot_idx'])[mask]
    paths_arr = np.asarray(traj['paths'])[mask]
    states    = np.asarray(traj['states'])[mask]
    amps      = np.asarray(traj['amplitudes'])[mask]
    pos       = traj['positions'][mask]
    vel       = traj['velocities'][mask]

    # Decide grouping mode
    max_path_len = max((len(p) for p in paths_arr), default=0)
    if arm_grouping == 'auto':
        use_arm = max_path_len > 10
    else:
        use_arm = bool(arm_grouping)

    # Assign group labels
    if use_arm:
        # Vectorised arm-key extraction — no Python loop over path strings.
        # Truncate every path to its first 2 Unicode characters (dtype='U2'),
        # then view as U1 to extract the character at index 1.
        # Paths with length 1 (initial wavepacket) get a null char (\x00) at
        # position 1 after truncation, which we map to 'init'.
        paths_2 = np.asarray(paths_arr, dtype='U2')            # C-level truncation
        chars   = paths_2.view('U1').reshape(-1, 2)[:, 1]      # O(1) view, no copy
        group_keys = np.where(chars == '\x00', 'init',
                     np.where(chars == '0',    'lower', 'upper'))
    else:
        group_keys = paths_arr

    g = 0.0 if potential in ('zero_pot', 'rotating_pot') else _G

    if rotation is None:
        rotation = traj.get('rotation', np.zeros(3))
    free_flight = _free_flight_factory(g, rotation)

    unique_groups = list(dict.fromkeys(group_keys))
    result = {}

    for gkey in unique_groups:
        gm_idx = np.where(group_keys == gkey)[0]   # indices of records in group

        # Sort records by snapshot index so we can group with reduceat (O(n log n))
        arm_snaps = snap_idx[gm_idx]
        order     = np.argsort(arm_snaps, kind='stable')
        arm_snaps = arm_snaps[order]
        arm_pos   = pos[gm_idx[order]]
        arm_vel   = vel[gm_idx[order]]
        arm_st    = states[gm_idx[order]]
        arm_amp   = amps[gm_idx[order]]

        # Group boundaries with np.unique (O(n))
        unique_si, first, counts = np.unique(arm_snaps,
                                             return_index=True,
                                             return_counts=True)

        # Sum per group with reduceat, then divide → vectorised mean (O(n))
        sum_pos = np.add.reduceat(arm_pos, first, axis=0)
        sum_vel = np.add.reduceat(arm_vel, first, axis=0)
        mean_pos = sum_pos / counts[:, None]
        mean_vel = sum_vel / counts[:, None]

        # Last-record state and amplitude (after final beamsplitter)
        last_in_group = first + counts - 1
        st_last  = int(arm_st[last_in_group[-1]])
        amp_last = float(arm_amp[last_in_group].mean())

        t_snap = snap_times[unique_si]
        x_snap, y_snap, z_snap   = mean_pos[:,0], mean_pos[:,1], mean_pos[:,2]
        vx_snap, vy_snap, vz_snap = mean_vel[:,0], mean_vel[:,1], mean_vel[:,2]

        # Base: all snapshot points — straight numpy, no Python loop
        t_out = list(t_snap)
        x_out = list(x_snap); y_out = list(y_snap); z_out = list(z_snap)

        # Only loop over intervals that need analytic interpolation (long gaps)
        if n_interp > 0:
            dts = np.diff(t_snap)
            for i in np.where(dts > min_interp_dt)[0]:
                p0 = [x_snap[i], y_snap[i], z_snap[i]]
                v0 = [vx_snap[i], vy_snap[i], vz_snap[i]]
                t_mid = np.linspace(t_snap[i], t_snap[i+1], n_interp + 2)[1:-1]
                xm, ym, zm = free_flight(p0, v0, t_snap[i], t_mid)
                t_out.extend(t_mid.tolist())
                x_out.extend(xm.tolist())
                y_out.extend(ym.tolist())
                z_out.extend(zm.tolist())

        t_arr = np.array(t_out); ord2 = np.argsort(t_arr)
        result[gkey] = {
            't':          t_arr[ord2],
            'x':          np.array(x_out)[ord2],
            'y':          np.array(y_out)[ord2],
            'z':          np.array(z_out)[ord2],
            'state':      st_last,
            'amplitude':  amp_last,
            'arm_grouped': use_arm,
        }

    return result


# ── Enclosed area and the Sagnac phase ────────────────────────────────────────

_AU     = 1.660539066e-27
_MASS   = 86.90888 * _AU        # kg, Sr-87
_HBAR   = 6.62607015e-34 / (2 * np.pi)


def arm_loop(traj_or_file, atom_idx=0, potential='linear_pot', n_interp=200,
             n_grid=2000, rotation=None):
    """
    Sample the closed circuit formed by the two interferometer arms.

    Both arms are interpolated onto a common time grid spanning the interval
    where they both exist (first beamsplitter to recombination), then the upper
    arm is traversed forwards and the lower arm backwards to close the loop.

    Returns
    -------
    dict with keys ``t`` (K,), ``upper`` (K, 3), ``lower`` (K, 3) and
    ``loop`` (2K+1, 3) — the closed circuit, first point repeated at the end.
    """
    if isinstance(traj_or_file, (str,)):
        traj = load_trajectory(traj_or_file)
    else:
        traj = traj_or_file

    smooth = reconstruct_trajectories(traj, atom_idx=atom_idx,
                                      potential=potential, n_interp=n_interp,
                                      arm_grouping=True, rotation=rotation)

    if 'upper' not in smooth or 'lower' not in smooth:
        raise ValueError("could not identify both arms; "
                         f"found groups {sorted(smooth)}")

    up, lo = smooth['upper'], smooth['lower']
    t_min = max(up['t'][0],  lo['t'][0])
    t_max = min(up['t'][-1], lo['t'][-1])
    if t_max <= t_min:
        raise ValueError("upper and lower arms do not overlap in time")

    t = np.linspace(t_min, t_max, n_grid)
    sample = lambda d: np.column_stack([np.interp(t, d['t'], d[k])
                                        for k in ('x', 'y', 'z')])
    upper, lower = sample(up), sample(lo)

    loop = np.vstack([upper, lower[::-1], upper[:1]])
    return {'t': t, 'upper': upper, 'lower': lower, 'loop': loop}


def enclosed_area(traj_or_file, **kwargs):
    """
    Vector area enclosed by the two interferometer arms, in m².

    Computed as the discrete form of A = ½ ∮ r × dr around the closed circuit,
    which is exact for a piecewise-linear loop and needs no assumption that the
    loop is planar.

    The Sagnac phase follows as Δφ = 2m Ω·A / ħ — see :func:`sagnac_phase`.

    Use the Ω = 0 trajectories
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    To predict the *first-order* rotation phase, evaluate this on the
    **non-rotating** run. Rotation deflects the two arms by different amounts,
    so an area taken from a rotating run picks up its own O(Ω) piece, and
    2mΩ·A/ħ then mixes in an O(Ω²) term.

    For a single loop that distinction is a ~0.1 % detail. For an even number of
    loops it dominates: the first-order area cancels between the
    oppositely-circulating loops, so what survives in a rotating run is almost
    entirely the O(Ω²) contamination, and the area law appears to fail by orders
    of magnitude. Compare against the odd-in-Ω part of the phase,
    ``(φ(+Ω) − φ(−Ω))/2``, which isolates the same order.
    """
    loop = arm_loop(traj_or_file, **kwargs)['loop']
    return 0.5 * np.cross(loop[:-1], loop[1:]).sum(axis=0)


def sagnac_phase(area, rotation):
    """
    Sagnac phase Δφ = 2m Ω·A / ħ [rad], for vector area *area* [m²] and frame
    angular velocity *rotation* [rad/s].
    """
    return 2.0 * _MASS * float(np.dot(rotation, area)) / _HBAR


# ── Plotting ──────────────────────────────────────────────────────────────────

_PULSE_COLOR = '#d62728'                      # reddish, shown at low alpha
_STATE_COLOR = {0: '#2ca02c', 1: '#1f77b4'}   # green=ground, blue=excited
_STATE_LABEL = {0: 'ground (s=0)', 1: 'excited (s=1)'}
_ARM_COLOR   = {'lower': '#2ca02c', 'upper': '#1f77b4', 'init': '#aaaaaa'}
_ARM_LABEL   = {'lower': 'lower arm', 'upper': 'upper arm', 'init': 'initial'}


def plot_trajectory(traj_or_file, atom_idx=0, potential='linear_pot',
                    n_interp=200, figsize=(9, 6), lw_scale=1.5,
                    show_pulses=True, show_transverse=True, z_unit='cm',
                    axes=None):
    """
    Plot interferometer spacetime diagram and transverse trajectory.

    Parameters
    ----------
    traj_or_file : str/Path or dict
        _TRAJ.h5 file path or output of :func:`load_trajectory`.
    atom_idx   : int   — atom to plot (default 0; trajectory mode → use 0)
    potential  : str   — ``'linear_pot'`` or ``'zero_pot'``
    n_interp   : int   — interpolation points per segment (default 200)
    figsize    : tuple — figure size
    lw_scale   : float — line-width multiplier (scales with amplitude)
    show_pulses: bool  — shade pulse intervals as reddish bands
    show_transverse : bool — also draw the transverse ($x$) subplot below
    z_unit     : str   — ``'cm'`` or ``'m'`` for the vertical ($z$) axis
    axes       : (ax_z, ax_x) or None — provide existing axes to draw into
        (``ax_x`` may be ``None`` when ``show_transverse=False``)

    Returns
    -------
    fig, (ax_z, ax_x)   — ``ax_x`` is ``None`` when ``show_transverse=False``
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if isinstance(traj_or_file, (str,)):
        traj = load_trajectory(traj_or_file)
    else:
        traj = traj_or_file

    smooth = reconstruct_trajectories(traj, atom_idx=atom_idx,
                                      potential=potential, n_interp=n_interp)

    z_scale = 100.0 if z_unit == 'cm' else 1.0

    if axes is None:
        if show_transverse:
            fig, (ax_z, ax_x) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                              gridspec_kw=dict(hspace=0.08))
        else:
            fig, ax_z = plt.subplots(figsize=figsize)
            ax_x = None
    else:
        ax_z, ax_x = axes
        fig = ax_z.get_figure()

    plot_axes = (ax_z, ax_x) if ax_x is not None else (ax_z,)

    # ── shade pulse intervals ─────────────────────────────────────────────────
    if show_pulses:
        snap_times  = traj['snapshot_times']
        snap_labels = traj['snapshot_labels']
        starts = {lbl: t for t, lbl in zip(snap_times, snap_labels)
                  if lbl.endswith('_start')}
        ends   = {lbl: t for t, lbl in zip(snap_times, snap_labels)
                  if lbl.endswith('_end')}
        for key, t_start in starts.items():
            pulse_id = key.replace('_start', '_end')
            if pulse_id in ends:
                for ax in plot_axes:
                    ax.axvspan(t_start, ends[pulse_id],
                               color=_PULSE_COLOR, alpha=0.12, lw=0)

    # ── draw trajectories ─────────────────────────────────────────────────────
    arm_grouped = any(d.get('arm_grouped', False) for d in smooth.values())
    seen_keys = set()
    for key, data in smooth.items():
        amp = data['amplitude']
        lw  = max(0.4, lw_scale * amp)
        if arm_grouped:
            col = _ARM_COLOR.get(key, 'gray')
        else:
            col = _STATE_COLOR.get(data['state'], 'gray')
        ax_z.plot(data['t'], data['z'] * z_scale,  color=col, lw=lw, alpha=0.85)
        if ax_x is not None:
            ax_x.plot(data['t'], data['x'] * 1e3, color=col, lw=lw, alpha=0.85)
        seen_keys.add(key)

    # ── labels ────────────────────────────────────────────────────────────────
    ax_z.set_ylabel(f'$z$ [{z_unit}]')
    if ax_x is not None:
        ax_x.set_ylabel('$x$ [mm]')
        ax_x.set_xlabel('$t$ [s]')
    else:
        ax_z.set_xlabel('$t$ [s]')

    if arm_grouped:
        legend_lines = [Line2D([0], [0], color=_ARM_COLOR[k], lw=1.5,
                                label=_ARM_LABEL[k])
                        for k in ('lower', 'upper') if k in seen_keys]
    else:
        seen_states = {d['state'] for d in smooth.values()}
        legend_lines = [Line2D([0], [0], color=_STATE_COLOR[s], lw=1.5,
                                label=_STATE_LABEL[s])
                        for s in sorted(seen_states)]
    if show_pulses:
        legend_lines.append(Patch(facecolor=_PULSE_COLOR, alpha=0.2,
                                   label='laser pulse'))
    ax_z.legend(handles=legend_lines, fontsize=8, loc='best')

    snap_times = traj['snapshot_times']
    t_det = snap_times[-1]
    ax_z.set_xlim(snap_times[0], t_det)
    ax_z.set_title(f'Interferometer trajectory  (atom {atom_idx})', fontsize=9)

    return fig, (ax_z, ax_x)


def _auto_scale(values):
    """Pick a metric prefix for an axis spanning *values*. Returns (factor, unit)."""
    span = float(np.nanmax(values) - np.nanmin(values)) if len(values) else 0.0
    for factor, unit in ((1.0, 'm'), (1e3, 'mm'), (1e6, '$\\mu$m'), (1e9, 'nm')):
        if span * factor >= 1.0:
            return factor, unit
    return 1e9, 'nm'


def plot_trajectory_planes(traj_or_file, atom_idx=0, potential='linear_pot',
                           n_interp=200, figsize=(13, 4.2), planes=('xy', 'xz', 'yz'),
                           shade_area=True, rotation=None, sagnac_rotation=None,
                           exaggerate='auto', axes=None, title=None):
    """
    Plot the two interferometer arms as position-space traces, projected onto
    the requested coordinate planes.

    This is the view that makes the Sagnac loop visible: the arms leave the
    first beamsplitter together, separate, and close again at recombination, so
    each projection is a closed loop whose signed area sets the rotation phase.
    A non-rotating run collapses the transverse projections to a line.

    Parameters
    ----------
    traj_or_file : str or dict — _TRAJ.h5 path or :func:`load_trajectory` output
    atom_idx   : int
    potential  : str   — see :func:`reconstruct_trajectories`
    n_interp   : int   — interpolation points per free-flight segment
    planes     : tuple — any of ``'xy'``, ``'xz'``, ``'yz'``
    shade_area : bool  — fill the enclosed loop and annotate its projected area
    exaggerate : ``'auto'``, or a float — factor by which the arm separation is
        blown up about the mean arm before drawing. The arms are typically ~10⁶
        times closer together than the trajectory is long, so at true scale
        (``exaggerate=1``) the loop collapses to a line. The factor is chosen
        per panel and stated in its title; the quoted areas are always the true
        ones.
    rotation   : (3,) array or None — the rate the run was *simulated* at, used
                 to reconstruct the free flight. ``None`` takes ``traj['rotation']``.
                 Overriding it to something the data was not generated with will
                 corrupt the interpolation between snapshots.
    sagnac_rotation : (3,) array or None — the rate used only to turn the
                 enclosed area into a phase. Defaults to *rotation*. Pass it
                 explicitly to predict the rotation phase from a non-rotating
                 run, which is the correct first-order prescription (see
                 :func:`enclosed_area`).
    axes       : sequence of axes matching *planes*, or None
    title      : str or None — figure suptitle

    Returns
    -------
    fig, axes, info
        *info* is a dict with ``area`` (the full 3-vector area, m²),
        ``rotation`` and ``sagnac_phase`` (rad).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if isinstance(traj_or_file, (str,)):
        traj = load_trajectory(traj_or_file)
    else:
        traj = traj_or_file

    rot = traj.get('rotation', np.zeros(3)) if rotation is None else np.asarray(rotation, float)
    sag = rot if sagnac_rotation is None else np.asarray(sagnac_rotation, float)

    loops = arm_loop(traj, atom_idx=atom_idx, potential=potential,
                     n_interp=n_interp, rotation=rot)
    area = enclosed_area(traj, atom_idx=atom_idx, potential=potential,
                         n_interp=n_interp, rotation=rot)

    if axes is None:
        fig, axes = plt.subplots(1, len(planes), figsize=figsize)
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].get_figure()

    idx = {'x': 0, 'y': 1, 'z': 2}
    # the area component normal to each plane, e.g. the xy projection encloses A_z
    normal = {'xy': 2, 'xz': 1, 'yz': 0}

    com = 0.5 * (loops['upper'] + loops['lower'])
    delta = loops['upper'] - loops['lower']

    # residual arm separation at recombination: zero only for a perfectly closed
    # interferometer, which a rotating frame does not give you for free
    gap_vec = delta[-1]
    gap = float(np.linalg.norm(gap_vec))
    factors = []

    for ax, plane in zip(axes, planes):
        i, j = idx[plane[0]], idx[plane[1]]

        # The arms are typically a million times closer together than the
        # trajectory is long, so at true scale the loop collapses onto a line.
        # Blow the separation up about the mean arm by a round factor, and say
        # so in the panel title.
        if exaggerate == 'auto':
            span = max(np.ptp(com[:, i]), np.ptp(com[:, j]))
            sep = max(np.max(np.abs(delta[:, i])), np.max(np.abs(delta[:, j])))
            if sep > 0 and span > 0:
                factor = 10.0 ** np.floor(np.log10(0.3 * span / sep))
            else:
                factor = 1.0
        else:
            factor = float(exaggerate)
        factor = max(factor, 1.0)
        factors.append(factor)

        up = com + 0.5 * factor * delta
        lo = com - 0.5 * factor * delta
        loop = np.vstack([up, lo[::-1], up[:1]])

        si, ui = _auto_scale(np.concatenate([up[:, i], lo[:, i]]))
        sj, uj = _auto_scale(np.concatenate([up[:, j], lo[:, j]]))

        if shade_area:
            ax.fill(loop[:, i] * si, loop[:, j] * sj,
                    color='#888888', alpha=0.18, lw=0, zorder=0)

        ax.plot(up[:, i] * si, up[:, j] * sj,
                color=_ARM_COLOR['upper'], lw=1.4, label=_ARM_LABEL['upper'])
        ax.plot(lo[:, i] * si, lo[:, j] * sj,
                color=_ARM_COLOR['lower'], lw=1.4, label=_ARM_LABEL['lower'])

        # mark the start (first beamsplitter) and end (recombination)
        ax.plot(up[0, i] * si, up[0, j] * sj, 'o', ms=4, color='k', zorder=5)
        ax.plot(up[-1, i] * si, up[-1, j] * sj, 's', ms=4, color='k', zorder=5)

        ax.set_xlabel(f'${plane[0]}$ [{ui}]')
        ax.set_ylabel(f'${plane[1]}$ [{uj}]')

        a_n = area[normal[plane]]
        exag = '' if factor == 1.0 else f'   (separation $\\times${factor:.0g})'
        ax.set_title(f'{plane[0]}–{plane[1]} plane   $A_{{{"xyz"[normal[plane]]}}}$ = '
                     f'{a_n:.3e} m$^2${exag}', fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)

    handles = [Line2D([0], [0], color=_ARM_COLOR[k], lw=1.5, label=_ARM_LABEL[k])
               for k in ('upper', 'lower')]
    handles += [Line2D([0], [0], marker='o', color='k', lw=0, ms=4, label='first BS'),
                Line2D([0], [0], marker='s', color='k', lw=0, ms=4, label='recombination')]
    axes[0].legend(handles=handles, fontsize=7, loc='best')

    dphi = sagnac_phase(area, sag)
    if title is None:
        title = (f'Interferometer arms in position space   '
                 f'$\\Omega$ = ({sag[0]:.3g}, {sag[1]:.3g}, {sag[2]:.3g}) rad/s   '
                 f'$\\Delta\\varphi_{{Sagnac}}$ = {dphi:.4g} rad   '
                 f'arm gap at recombination = {gap:.3g} m')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    return fig, axes, {'area': area, 'rotation': sag, 'sagnac_phase': dphi,
                       'gap': gap, 'gap_vector': gap_vec,
                       'max_separation': float(np.max(np.linalg.norm(delta, axis=1))),
                       'exaggeration': factors}


def plot_loop_comparison(entries, potential='linear_pot', n_interp=200,
                         plane='xz', figsize=None, rotation=None,
                         sagnac_rotation=None):
    """
    Compare the enclosed loops of several sequences side by side — the intended
    use is 1-, 2- and 4-loop interferometers, where the successive loops are
    traversed in opposite senses so the signed areas cancel pairwise and the
    rotation phase collapses for even loop counts.

    Parameters
    ----------
    entries : sequence of (label, traj_or_file)
    plane   : str  — which projection to draw (default ``'xz'``)
    sagnac_rotation : (3,) array or None — Ω used only for the area→phase
        conversion. Pass the intended rotation rate here and feed non-rotating
        trajectories in *entries*: that is the correct first-order prescription,
        and for even loop counts it is the only one that gives the right answer
        (see :func:`enclosed_area`).

    Returns
    -------
    fig, axes, rows
        *rows* is a list of dicts with ``label``, ``area``, ``sagnac_phase``.
    """
    import matplotlib.pyplot as plt

    n = len(entries)
    if figsize is None:
        figsize = (4.3 * n, 4.2)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    axes = np.atleast_1d(axes)

    rows = []
    for ax, (label, src) in zip(axes, entries):
        traj = load_trajectory(src) if isinstance(src, str) else src
        rot = traj.get('rotation', np.zeros(3)) if rotation is None else np.asarray(rotation, float)
        _, _, info = plot_trajectory_planes(traj, potential=potential,
                                            n_interp=n_interp, planes=(plane,),
                                            rotation=rot,
                                            sagnac_rotation=sagnac_rotation,
                                            axes=[ax], title='')
        dphi = info['sagnac_phase']
        # keep the exaggeration factor visible: without it the arms look metres
        # apart when they are in fact millimetres apart
        ax.set_title(f'{label}\n$\\Delta\\varphi_{{Sagnac}}$ = {dphi:.3e} rad\n'
                     f'max arm sep. {info["max_separation"]*1e3:.2f} mm, '
                     f'drawn $\\times${info["exaggeration"][0]:.0g}',
                     fontsize=9)
        rows.append({'label': label, 'area': info['area'], 'sagnac_phase': dphi,
                     'gap': info['gap'],
                     'max_separation': info['max_separation'],
                     'exaggeration': info['exaggeration'][0]})

    fig.suptitle('Enclosed area vs number of loops — successive loops reverse '
                 'the circulation, so even loop counts cancel', fontsize=10)
    fig.tight_layout()
    return fig, axes, rows


def plot_arm_separation(traj_or_file, atom_idx=0, potential='linear_pot',
                        n_interp=200, figsize=(7, 3)):
    """
    Plot the vertical separation Δz between upper and lower interferometer arms
    versus time, revealing the enclosed spacetime area.

    Works only when exactly two paths have the same state (e.g. the two
    ground-state arms before recombination).
    """
    import matplotlib.pyplot as plt

    if isinstance(traj_or_file, str):
        traj = load_trajectory(traj_or_file)
    else:
        traj = traj_or_file

    smooth = reconstruct_trajectories(traj, atom_idx=atom_idx,
                                      potential=potential, n_interp=n_interp)

    # Find pairs of same-state paths (the two MZ arms)
    from collections import defaultdict
    by_state = defaultdict(list)
    for path, data in smooth.items():
        by_state[data['state']].append((path, data))

    fig, ax = plt.subplots(figsize=figsize)
    colors = {0: '#2ca02c', 1: '#d62728'}

    for state, pairs in by_state.items():
        if len(pairs) < 2:
            continue
        # Interpolate both paths to a common time grid, then diff
        t_common = pairs[0][1]['t']   # use first path's time grid as reference
        for (p1, d1), (p2, d2) in zip(pairs[:-1], pairs[1:]):
            # Only meaningful while both paths exist at the same times
            t_min = max(d1['t'][0], d2['t'][0])
            t_max = min(d1['t'][-1], d2['t'][-1])
            if t_max <= t_min:
                continue
            t_grid = np.linspace(t_min, t_max, 500)
            z1 = np.interp(t_grid, d1['t'], d1['z'])
            z2 = np.interp(t_grid, d2['t'], d2['z'])
            dz = np.abs(z1 - z2)
            ax.plot(t_grid, dz * 100, color=colors[state], lw=1.2,
                    label=f'state {state}: |{p1} − {p2}|')

    ax.set_xlabel('$t$ [s]')
    ax.set_ylabel('Arm separation $|\\Delta z|$ [cm]')
    ax.set_title('MZ arm separation vs time')
    ax.legend(fontsize=8)
    ax.set_xlim(traj['snapshot_times'][0], traj['snapshot_times'][-1])
    plt.tight_layout()
    return fig, ax
