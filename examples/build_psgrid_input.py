"""
build_psgrid_input.py — generate a psgrid ais++ input file for surrogate training.

Writes a single .aisi file using ``initmode psgrid``.  One atom is simulated
per phase-space grid node (x0, vx0), giving the full transfer function
p(port | x0, vx0) without any Gaussian ensemble sampling.  The output
PSMAP can then be loaded into ``aispy.psmap.PSMAPSurrogate`` to generate
synthetic shots without further ais++ runs.

The sequence is the same n=1 Mach-Zehnder used in PSR_EXAMPLE_NLMT1.aisi
(T = 2.225 s, Sr-87 clock transition, Gaussian beam).  The phase-shear
kick (kx_psr) is set to zero here; any phase shear is applied offline by
the surrogate.

Usage
-----
    cd examples/
    python build_psgrid_input.py [options]

    # then run ais++:
    ais++ -i input-files/PSGRID.aisi -o output-files/PSGRID.h5

Options
-------
  --nx, --nvx          Grid points in x0 and vx0 (default: 25)
  --sigma_x            ±N_sigma half-width to cover in x0 (default: 3)
  --sigma_vx           ±N_sigma half-width to cover in vx0 (default: 3)
  --cloud_sigma_x_m    Cloud 1σ position spread [m] (default: 100e-6)
  --cloud_T_nK         Cloud transverse temperature [nK] (default: 1)
  --stem               Output file stem (default: PSGRID)
"""

import argparse
import math
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from aispy.utils import AISFlow, pi, hbar, kz

# ── fixed interferometer parameters (match PSR_EXAMPLE_NLMT1.aisi) ───────────
T_INTERROG     = mp.mpf('2.225')           # s
RABI_FREQ      = 2 * pi * mp.mpf('1e3')   # rad/s
DT_LMT         = mp.mpf('0.0')            # s  (n=1: no LMT sub-pulses)
DETECTION_TIME = 2 * T_INTERROG + mp.mpf('0.001')
BEAM_WAIST     = mp.mpf('0.01000319328186729550325188260259019')  # m
BEAM_RADIUS    = mp.mpf('0.01')           # m
BASELINE       = mp.mpf('10')             # m

# physical constants
kB     = 1.380649e-23   # J/K
M_SR87 = 86.909 * 1.66054e-27  # kg


def sigma_vx_from_T(T_K):
    """Transverse velocity spread [m/s] from temperature [K]."""
    return math.sqrt(kB * T_K / M_SR87)


def build_param_dict(xgrid, ygrid, zgrid, vxgrid, vygrid, vzgrid,
                     cloud_sigma_x, cloud_T_K):
    """Return the full ais++ parameter dictionary for a psgrid run."""
    vz0 = vzgrid[0]   # all atoms have the same vz0 in a 1D psgrid
    return {
        'cloud_params': {
            # psgrid init mode
            'initmode':     'psgrid',
            'xgrid':        xgrid,
            'ygrid':        ygrid,
            'zgrid':        zgrid,
            'vxgrid':       vxgrid,
            'vygrid':       vygrid,
            'vzgrid':       vzgrid,
            # Gaussian params below are required by the ais++ parser but
            # are ignored when initmode=psgrid.
            'natoms':       1,
            'initialstate': 0,
            'sigma':        cloud_sigma_x,
            'transtemp':    cloud_T_K,
            'longtemp':     0,
            'x0':           [0.0, 0.0, 0.0],
            'v0':           [0.0, 0.0, float(vz0)],
        },
        'potential_params': {
            'utype': 'linear_pot',
        },
        'sequence_params': {
            't_init':             mp.mpf('0.0'),
            'detectiontime':      DETECTION_TIME,
            'interrogation_time': [T_INTERROG],
            'lmt_order':          1,        # n=1: simple MZ, no LMT blocks
            'dt_lmt':             DT_LMT,
            'automaticdetuning':  1,
            'frequencychirp':     0,
            'kchirp':             0,
            'ultranarrow':        True,
            'sequencename':       'MZ',
            'loopnumber':         1,
        },
        'pulse_params': {
            'rabi_freq':      RABI_FREQ,
            'wtype':          'gaussian',
            'phi0':           0,
            # kx_psr=0: no shear baked in — apply offline via surrogate
            'kx_psr':         0,
            'ky_psr':         0,
            'waist':          BEAM_WAIST,
            'focallength':    0,
            'zupwardlaser':   0,
            'zdownwardlaser': 0,
            'beam_radius':    BEAM_RADIUS,
            'baseline':       BASELINE,
            'zernike_params': {},
        },
        'simulation_params': {
            'amplitudethreshold': 0,
            'coherencelength':    float(3.740463112189228e-06),
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nx',             type=int,   default=25,
                   help='grid points in x0 (default: 25)')
    p.add_argument('--nvx',            type=int,   default=25,
                   help='grid points in vx0 (default: 25)')
    p.add_argument('--sigma_x',        type=float, default=3.0,
                   help='half-width in units of cloud σ_x (default: 3)')
    p.add_argument('--sigma_vx',       type=float, default=3.0,
                   help='half-width in units of cloud σ_vx (default: 3)')
    p.add_argument('--cloud_sigma_x_m', type=float, default=100e-6,
                   help='cloud 1σ position spread [m] (default: 100e-6)')
    p.add_argument('--cloud_T_nK',     type=float, default=1.0,
                   help='cloud transverse temperature [nK] (default: 1)')
    p.add_argument('--vz0',            type=float, default=19.62,
                   help='launch velocity vz0 [m/s] (default: 19.62)')
    p.add_argument('--stem',           type=str,   default='PSGRID',
                   help='output file stem (default: PSGRID)')
    args = p.parse_args()

    cloud_T_K  = args.cloud_T_nK * 1e-9
    sig_x      = args.cloud_sigma_x_m
    sig_vx     = sigma_vx_from_T(cloud_T_K)

    x_half  = args.sigma_x  * sig_x
    vx_half = args.sigma_vx * sig_vx

    xgrid  = [-x_half,  x_half,  args.nx]
    ygrid  = [0.0,      0.0,     1]
    zgrid  = [0.0,      0.0,     1]
    vxgrid = [-vx_half, vx_half, args.nvx]
    vygrid = [0.0,      0.0,     1]
    vzgrid = [args.vz0, args.vz0, 1]

    n_atoms = args.nx * args.nvx
    t_det   = float(2 * T_INTERROG + mp.mpf('0.001'))

    print(f'Grid             : {args.nx} × {args.nvx} = {n_atoms} atoms')
    print(f'x0  range        : {-x_half*1e6:.1f} … {x_half*1e6:.1f} µm')
    print(f'vx0 range        : {-vx_half*1e3:.2f} … {vx_half*1e3:.2f} mm/s')
    print(f'vz0              : {args.vz0} m/s')
    print(f'Detection time   : {t_det:.4f} s')
    print()

    outdir = os.path.join(os.path.dirname(__file__), 'input-files')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output-files'),
                exist_ok=True)

    param_dict = build_param_dict(xgrid, ygrid, zgrid, vxgrid, vygrid, vzgrid,
                                   sig_x, cloud_T_K)
    AISFlow(param_dict, args.stem, outdir)

    aisi_path  = os.path.join(outdir, args.stem + '.aisi')
    output_h5  = os.path.join(os.path.dirname(__file__),
                               'output-files', args.stem + '.h5')
    print(f'Written: {aisi_path}')
    print()
    print('Run ais++ with:')
    print(f'  ais++ -i {aisi_path} \\')
    print(f'        -o {output_h5}')
    print()
    print('Then load the surrogate in Python:')
    print(f'  from aispy.psmap import load_psmap, PSMAPSurrogate')
    print(f'  sur = PSMAPSurrogate(load_psmap("{output_h5}"), t_det={t_det:.4f})')


if __name__ == '__main__':
    main()
