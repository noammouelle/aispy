"""
build_inputs.py — generate ais++ input files for the wavefront-aberration example.

Uses aispy.utils.AISFlow to build .aisi files for a 5×5 grid of initial
cloud positions and velocities (25 simulations total), matching the
Phase-Shear Readout Mach–Zehnder sequence used in [Mouelle et al. 2025].

Usage
-----
    cd examples/
    python build_inputs.py --natoms 100000 --nlmt 101

This writes 25 files to  examples/input-files/
named  PSR_WA_NLMT<n>_VX0<vx0>_X0<x0>.aisi

After building, run each simulation with:
    ais++ -i input-files/PSR_WA_NLMT101_VX00.000e+00_X00.000e+00.aisi \
          -o output-files/PSR_WA_NLMT101_VX00.000e+00_X00.000e+00.h5
or use a batch runner (e.g. GNU parallel or a bash loop) for all 25 files.
"""

import argparse
import os
import sys
import mpmath as mp

# ── ensure aispy is importable when running from examples/ ───────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from aispy.utils import AISFlow, pi, hbar, kz

# ── interferometer constants ──────────────────────────────────────────────────
INTERROGATION_TIME = mp.mpf('2.225')        # s
RABI_FREQ          = 2 * pi * mp.mpf('1e3') # rad/s
TAU                = mp.pi / (RABI_FREQ * 2)
DT_LMT             = mp.mpf('1e-7')         # s  (LMT sub-pulse spacing)
DETECTION_TIME     = 2 * INTERROGATION_TIME + mp.mpf('0.001')
ZR                 = 450.085               # m  (Rayleigh range, w0 = 3 cm)

# ── grid of initial conditions ────────────────────────────────────────────────
VX0_LIST = [-2e-4, -1e-4, 0.0, 1e-4, 2e-4]  # m/s  (transverse COM velocity)
X0_LIST  = [-1e-3, -5e-4, 0.0, 5e-4, 1e-3]  # m    (transverse COM position)


def build_param_dict(x0, vx0, natoms, lmt_order):
    """Return the full ais++ parameter dictionary for one (x0, vx0) pair."""
    return {
        'cloud_params': {
            'natoms':       natoms,
            'initialstate': 0,
            'sigma':        100e-6,   # transverse cloud radius (m)
            'longtemp':     0,
            'transtemp':    1e-9,     # 1 nK
            'x0':           [x0, 0.0, 0.0],
            'v0':           [vx0, 0.0, 19.62],
        },
        'potential_params': {
            'utype': 'linear_pot',
        },
        'sequence_params': {
            't_init':            mp.mpf('0.0'),
            'detectiontime':     DETECTION_TIME,
            'interrogation_time': [INTERROGATION_TIME],
            'lmt_order':         lmt_order,
            'dt_lmt':            DT_LMT * (lmt_order != 1),
            'automaticdetuning': 1,
            'frequencychirp':    0,
            'kchirp':            0,
            'ultranarrow':       True,
            'sequencename':      'MZ',
            'loopnumber':        1,
        },
        'pulse_params': {
            'rabi_freq':      RABI_FREQ,
            'wtype':          'gaussian',
            'phi0':           0,
            'kx_psr':         3140,   # rad/m  (Phase-Shear Readout kick)
            'ky_psr':         0,
            'ky':             0,
            'waist':          mp.sqrt(2 * ZR / kz),
            'focallength':    0,
            'zupwardlaser':   0,
            'zdownwardlaser': 0,
            'beam_radius':    0.02,
            'baseline':       10,
            'zernike_params': {
                '1': [0, 0, 0, 0, 0, 0],
                '4': [0, 0, 0, 0, 0, 0],
            },
        },
        'simulation_params': {
            'amplitudethreshold': 0,
            'coherencelength':    10 * 0.5 * float(hbar) / (
                                      (1.44e-25 * 1.38e-23 * 10e-9) ** 0.5),
            'usemcbranching':     0,
            'ignoredetuning':     0,
            'usestaticapprox':    0,
            'seed':               -1,
            'usedetvolselection': 0,
            'usepathselection':   1,
            'xdet':               [-3e-2, 3e-2],
            'ydet':               [-3e-2, 3e-2],
            'zdet':               [-3e-2, 3e-2],
            'gslqagabserr':       1e-12,
            'gslqagrelerr':       1e-12,
            'gslkinodeabserr':    1e-9,
            'gslkinoderelerr':    0,
            'gslpulseodeabserr':  1e-9,
            'gslpulseoderelerr':  0,
            'ultrafast':          1,
        },
        'io_params': {
            'printprobs':       0,
            'printwavepackets': 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build ais++ input files for the wavefront-aberration PSR example.')
    parser.add_argument('--natoms', type=int, default=100_000,
                        help='atoms per simulation (default: 100000)')
    parser.add_argument('--nlmt',   type=int, default=101,
                        help='LMT order (default: 101)')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'input-files')
    os.makedirs(outdir, exist_ok=True)

    total = len(VX0_LIST) * len(X0_LIST)
    print(f'Building {total} .aisi files  (natoms={args.natoms}, nlmt={args.nlmt})')
    print(f'Output directory: {outdir}\n')

    count = 0
    for vx0 in VX0_LIST:
        for x0 in X0_LIST:
            stem    = f'PSR_WA_NLMT{args.nlmt}_VX0{vx0:.3e}_X0{x0:.3e}'
            workdir = outdir
            flowdir = stem   # AISFlow writes  workdir/flowdir.aisi

            param_dict = build_param_dict(x0, vx0, args.natoms, args.nlmt)
            AISFlow(param_dict, flowdir, workdir)

            count += 1
            print(f'  [{count:2d}/{total}]  {stem}.aisi')

    print(f'\nDone.  Run each simulation with:')
    print(f'  ais++ -i input-files/<stem>.aisi -o output-files/<stem>.h5')


if __name__ == '__main__':
    main()
