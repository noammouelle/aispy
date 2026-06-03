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
        velocities      (N, 3)    float64 [m/s]
    """
    with h5py.File(fname, 'r') as f:
        decode = lambda arr: [x.decode() if isinstance(x, bytes) else x
                              for x in arr]
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
        }


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
                              arm_grouping='auto'):
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
    potential      : str    — ``'linear_pot'`` or ``'zero_pot'``
    n_interp       : int    — interpolation points per free-flight segment
    min_interp_dt  : float  — minimum interval [s] to interpolate (default 5 ms)
    arm_grouping   : str or bool — ``'auto'``, ``True``, or ``False``

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

    g = _G if potential == 'linear_pot' else 0.0

    def free_flight(p0, v0, t0, t_arr):
        dt = t_arr - t0
        return (p0[0] + v0[0]*dt,
                p0[1] + v0[1]*dt,
                p0[2] + v0[2]*dt - 0.5*g*dt**2)

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


# ── Plotting ──────────────────────────────────────────────────────────────────

_STATE_COLOR = {0: '#2ca02c', 1: '#d62728'}   # green=ground, red=excited
_STATE_LABEL = {0: 'ground (s=0)', 1: 'excited (s=1)'}
_ARM_COLOR   = {'lower': '#2ca02c', 'upper': '#d62728', 'init': '#aaaaaa'}
_ARM_LABEL   = {'lower': 'lower arm', 'upper': 'upper arm', 'init': 'initial'}


def plot_trajectory(traj_or_file, atom_idx=0, potential='linear_pot',
                    n_interp=200, figsize=(9, 6), lw_scale=1.5,
                    show_pulses=True, axes=None):
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
    show_pulses: bool  — shade pulse intervals as grey bands
    axes       : (ax_z, ax_x) or None — provide existing axes to draw into

    Returns
    -------
    fig, (ax_z, ax_x)
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

    if axes is None:
        fig, (ax_z, ax_x) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                          gridspec_kw=dict(hspace=0.08))
    else:
        ax_z, ax_x = axes
        fig = ax_z.get_figure()

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
                for ax in (ax_z, ax_x):
                    ax.axvspan(t_start, ends[pulse_id],
                               color='#aaaaaa', alpha=0.25, lw=0)

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
        ax_z.plot(data['t'], data['z'] * 100,  color=col, lw=lw, alpha=0.85)
        ax_x.plot(data['t'], data['x'] * 1e3, color=col, lw=lw, alpha=0.85)
        seen_keys.add(key)

    # ── labels ────────────────────────────────────────────────────────────────
    ax_z.set_ylabel('$z$ [cm]')
    ax_x.set_ylabel('$x$ [mm]')
    ax_x.set_xlabel('$t$ [s]')

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
        legend_lines.append(Patch(facecolor='#aaaaaa', alpha=0.35,
                                   label='laser pulse'))
    ax_z.legend(handles=legend_lines, fontsize=8, loc='best')

    snap_times = traj['snapshot_times']
    t_det = snap_times[-1]
    ax_z.set_xlim(snap_times[0], t_det)
    ax_z.set_title(f'Interferometer trajectory  (atom {atom_idx})', fontsize=9)

    return fig, (ax_z, ax_x)


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
