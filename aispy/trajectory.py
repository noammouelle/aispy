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

def reconstruct_trajectories(traj, atom_idx=0, potential='linear_pot',
                              n_interp=200, min_interp_dt=0.005):
    """
    Reconstruct smooth (t, x, y, z) curves from snapshot data.

    Between consecutive snapshots where the same path exists, the trajectory
    is obtained by evaluating the analytic free-flight equation at *n_interp*
    intermediate times.  For ``linear_pot`` this is exact.

    Interpolation is only applied when the time gap between snapshots exceeds
    *min_interp_dt* (default 5 ms).  This means:

    - **Free-flight intervals** (gap ~ T ~ seconds): interpolated with
      ``n_interp`` points → smooth parabolic arcs.
    - **LMT pulse intervals** (gap ~ 0.5 ms): not interpolated — the dense
      snapshot data is already sufficient to show the LMT structure.

    This avoids generating millions of redundant points for large-n runs
    while still rendering smooth free-flight segments.

    Parameters
    ----------
    traj           : dict — output of :func:`load_trajectory`
    atom_idx       : int  — which atom to reconstruct (default 0)
    potential      : str  — ``'linear_pot'`` (uniform gravity) or ``'zero_pot'``
    n_interp       : int  — interpolation points per free-flight segment (default 200)
    min_interp_dt  : float — minimum interval [s] to trigger interpolation (default 5 ms)

    Returns
    -------
    dict keyed by path string, each value a dict with:
        t          (M,) float64  — time [s]
        x, y, z    (M,) float64  — position [m]
        state       int          — internal state (0 or 1)
        amplitude   float64      — final wavepacket amplitude
    """
    snap_times = traj['snapshot_times']
    mask = np.asarray(traj['atom_indices']) == atom_idx

    snap_idx  = np.asarray(traj['snapshot_idx'])[mask]
    paths_arr = np.asarray(traj['paths'])[mask]
    states    = np.asarray(traj['states'])[mask]
    amps      = np.asarray(traj['amplitudes'])[mask]
    pos       = traj['positions'][mask]
    vel       = traj['velocities'][mask]

    g = _G if potential == 'linear_pot' else 0.0

    def free_flight(p0, v0, t0, t_arr):
        dt = t_arr - t0
        x = p0[0] + v0[0] * dt
        y = p0[1] + v0[1] * dt
        z = p0[2] + v0[2] * dt - 0.5 * g * dt**2
        return x, y, z

    unique_paths = list(dict.fromkeys(paths_arr))   # preserve encounter order
    result = {}

    for path in unique_paths:
        pm = paths_arr == path
        sn = snap_idx[pm]
        order = np.argsort(sn)
        sn   = sn[order]
        pp   = pos[pm][order]
        vv   = vel[pm][order]
        st   = states[pm][order]
        aa   = amps[pm][order]

        t_out, x_out, y_out, z_out = [], [], [], []

        for i, si in enumerate(sn):
            t0 = snap_times[si]
            p0 = pp[i];  v0 = vv[i]

            # Always record the snapshot point itself
            t_out.append(t0)
            x_out.append(p0[0]); y_out.append(p0[1]); z_out.append(p0[2])

            # If there is a next snapshot for this path, interpolate when
            # the interval is long enough (free-flight) but skip for short
            # LMT-pulse intervals where snapshots are already dense.
            if i < len(sn) - 1:
                t1 = snap_times[sn[i + 1]]
                dt = t1 - t0
                if dt > min_interp_dt and n_interp > 0:
                    t_mid = np.linspace(t0, t1, n_interp + 2)[1:-1]
                    xm, ym, zm = free_flight(p0, v0, t0, t_mid)
                    t_out.extend(t_mid.tolist())
                    x_out.extend(xm.tolist())
                    y_out.extend(ym.tolist())
                    z_out.extend(zm.tolist())

        # Sort by time (interleaved inserts may be out of order)
        t_out = np.array(t_out); x_out = np.array(x_out)
        y_out = np.array(y_out); z_out = np.array(z_out)
        ord2 = np.argsort(t_out)

        result[path] = {
            't':         t_out[ord2],
            'x':         x_out[ord2],
            'y':         y_out[ord2],
            'z':         z_out[ord2],
            'state':     int(st[-1]),
            'amplitude': float(aa[-1]),
        }

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

_STATE_COLOR = {0: '#2ca02c', 1: '#d62728'}   # green=ground, red=excited
_STATE_LABEL = {0: 'ground (s=0)', 1: 'excited (s=1)'}


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
    seen_states = set()
    for path, data in smooth.items():
        st  = data['state']
        amp = data['amplitude']
        col = _STATE_COLOR.get(st, 'gray')
        lw  = max(0.4, lw_scale * amp)
        ax_z.plot(data['t'], data['z'] * 100,  color=col, lw=lw, alpha=0.85)
        ax_x.plot(data['t'], data['x'] * 1e3, color=col, lw=lw, alpha=0.85)
        seen_states.add(st)

    # ── labels ────────────────────────────────────────────────────────────────
    ax_z.set_ylabel('$z$ [cm]')
    ax_x.set_ylabel('$x$ [mm]')
    ax_x.set_xlabel('$t$ [s]')

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
