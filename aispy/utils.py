from mpmath import mp, sqrt, cos, cosh, sin, sinh
import numpy as np
import datetime
import matplotlib.pyplot as plt

mp.dps = 34  # Set decimal precision to 34 digits

# define constants using mpmath for quad precision
c = mp.mpf('299792458')
g = mp.mpf('9.81')
R = mp.mpf('6.37e6')
au= mp.mpf('1.660539066e-27')
m = 86.90888 * au
h = mp.mpf('6.62607015e-34')
kB = 1.381e-23
pi = mp.pi
hbar = h / (2 * pi)

omega0 = 2 * pi * mp.mpf("429228004229873.0")
kz     = omega0 / c

def v_uniform(t,v0):
    return v0 - g*t

def v(t,v0,z0):
    return -1/2 * (R - 2*z0) * sqrt(2*g/R) * sinh(sqrt(2*g/R) * t) + v0 * cosh(sqrt(2*g/R) * t)

#def detuning(v,dummy):
#    omega = omega0 / (1 - v/c)
#    return omega

def detuning(v,e_to_g=False):
    # if e->g transition, need to use the negative sign
    if e_to_g == True:
        sign = -1
    else:
        sign = 1

    return sign*m*c**2/hbar * (1 -v/c - mp.sqrt((1-v/c)**2 - sign * 2*hbar/(m*c**2)*omega0))

class AISFlow():
    def __init__(self, param_dict, flowdir, workdir):
        self.param_dict = param_dict
        self.cloud_params = param_dict['cloud_params']
        self.potential_params = param_dict['potential_params']
        self.sequence_params = param_dict['sequence_params']
        self.pulse_params = param_dict['pulse_params']
        self.simulation_params = param_dict['simulation_params']
        self.io_params = param_dict['io_params']

        # open the aisi file
        self.aisi_file = open(workdir+'/'+flowdir+'.aisi', 'w')

        self._write_header()
        self._write_cloud_params()
        self._write_potential_params()
        self._write_simulation_params()   
        self._write_sequence_params()
        self._write_io_params()
        self._write_pulse_params()        
        
        # close the aisi file
        self.aisi_file.close()

    def _write_header(self):
        self.aisi_file.write('#     ___    _________           \n')
        self.aisi_file.write('#    /   |  /  _/ ___/  __    __  \n')
        self.aisi_file.write('#   / /| |  / / \__ \__/ /___/ /_ \n')
        self.aisi_file.write('#  / ___ |_/ / ___/ /_  __/_  __/ \n')
        self.aisi_file.write('# /_/  |_/___//____/ /_/   /_/    \n')
        self.aisi_file.write('#\n')
        self.aisi_file.write('#\n')
        self.aisi_file.write('# AISI file\n')
        self.aisi_file.write('# Created by ASIFlow\n')
        self.aisi_file.write('# Created on {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.aisi_file.write('\n')

    def _write_cloud_params(self):
        self.aisi_file.write('# Cloud parameters\n')
        initmode = self.cloud_params.get('initmode', 'gaussian')
        if initmode == 'psgrid':
            self.aisi_file.write('initmode psgrid\n')
            for axis in ('x', 'y', 'z', 'vx', 'vy', 'vz'):
                lo, hi, n = self.cloud_params[f'{axis}grid']
                self.aisi_file.write(f'{axis}grid {lo} {hi} {n}\n')
        # Gaussian params are always written — the ais++ parser requires them
        # even in psgrid mode (they are ignored in that mode).
        self.aisi_file.write('natoms {}\n'.format(self.cloud_params['natoms']))
        self.aisi_file.write('initialstate {}\n'.format(self.cloud_params['initialstate']))
        self.aisi_file.write('sigma {}\n'.format(self.cloud_params['sigma']))
        self.aisi_file.write('transtemp {}\n'.format(self.cloud_params['transtemp']))
        self.aisi_file.write('longtemp {}\n'.format(self.cloud_params['longtemp']))
        self.aisi_file.write('x0 {} {} {}\n'.format(self.cloud_params['x0'][0], self.cloud_params['x0'][1], self.cloud_params['x0'][2]))
        self.aisi_file.write('v0 {} {} {}\n'.format(self.cloud_params['v0'][0], self.cloud_params['v0'][1], self.cloud_params['v0'][2]))
        self.aisi_file.write('\n')

    def _write_potential_params(self):
        self.aisi_file.write('# Potential parameters\n')
        self.aisi_file.write('utype {}\n'.format(self.potential_params['utype']))
        # Frame angular velocity, rad/s. Only meaningful for the rotating_*
        # potentials; ais++ rejects it for the inertial ones.
        rotation = self.potential_params.get('rotation')
        if rotation is not None:
            self.aisi_file.write('rotation {} {} {}\n'.format(*rotation))
        self.aisi_file.write('\n')

    def _write_simulation_params(self):
        f = self.aisi_file
        s = self.simulation_params
        q = self.sequence_params

        f.write('# Simulation parameters\n')
        f.write(f'amplitudethreshold {s["amplitudethreshold"]}\n')
        f.write(f'coherencelength {s["coherencelength"]}\n')
        f.write(f'usemcbranching {s["usemcbranching"]}\n')
        f.write(f'usepathselection {s["usepathselection"]}\n')
        f.write(f'usestaticapprox {s["usestaticapprox"]}\n')
        f.write(f'ultrafast {s["ultrafast"]}\n')

        # --- pathstosimulate ---
        f.write('pathstosimulate ')
        pts = self._build_paths_to_simulate(self.sequence_params["loopnumber"],self.sequence_params["lmt_order"])
        for p in pts:
            self.aisi_file.write(p + " ")
        f.write('\n')

        f.write(f'ignoredetuning {s["ignoredetuning"]}\n')
        f.write(f'seed {s["seed"]}\n')
        f.write(f'usedetvolselection {s["usedetvolselection"]}\n')
        f.write(f'xdet {s["xdet"][0]} {s["xdet"][1]}\n')
        f.write(f'ydet {s["ydet"][0]} {s["ydet"][1]}\n')
        f.write(f'zdet {s["zdet"][0]} {s["zdet"][1]}\n')
        f.write(f'gslqagabserr {s["gslqagabserr"]}\n')
        f.write(f'gslqagrelerr {s["gslqagrelerr"]}\n')
        f.write(f'gslkinodeabserr {s["gslkinodeabserr"]}\n')
        f.write(f'gslkinoderelerr {s["gslkinoderelerr"]}\n')
        f.write(f'gslpulseodeabserr {s["gslpulseodeabserr"]}\n')
        f.write(f'gslpulseoderelerr {s["gslpulseoderelerr"]}\n\n')

    def _write_sequence_params(self):
        self.aisi_file.write('# Sequence parameters\n')
        self.aisi_file.write('detectiontime {}\n\n'.format(self.sequence_params['detectiontime']))

    def _write_io_params(self):
        self.aisi_file.write('# IO parameters\n')
        self.aisi_file.write('printprobs {}\n'.format(self.io_params['printprobs']))
        self.aisi_file.write('printwavepackets {}\n'.format(self.io_params['printwavepackets']))
        if self.io_params.get('printtrajectory', 0):
            self.aisi_file.write('printtrajectory 1\n')
        self.aisi_file.write('\n')

    def _write_pulse_params(self):
        if self.sequence_params['sequencename'] == 'MZ':
            if self.sequence_params['automaticdetuning'] == 1:
                if self.sequence_params['ultranarrow'] == 1:
                    #self._write_auto_stepwise_detuning_ultranarrow_MZ()
                    self._write_ultranarrow_MZ_Lloops()
                else:
                    self._write_auto_stepwise_detuning_MZ()
            else:
                if (self.sequence_params['frequencychirp'] != 0) or (self.sequence_params['kchirp'] != 0):
                    if self.sequence_params['ultranarrow'] == 1:
                        self._write_chirped_sequence_ultranarrow_MZ()
                    else:
                        self._write_chirped_sequence_MZ()
                else:
                    raise ValueError("Only chirped or automatically detuned sequences are supported at the moment")
                
        else:
            # 'RB' used to be dispatched here to _write_auto_stepwise_detuning_ultranarrow_RB
            # and _write_chirped_sequence_ultranarrow_RB, neither of which was ever
            # defined -- so selecting it raised AttributeError rather than anything
            # informative. Removed in favour of an honest error.
            raise ValueError(
                f"Unknown sequencename {self.sequence_params['sequencename']!r}. "
                f"Only 'MZ' is implemented."
            )
        
    def _build_paths_to_simulate(self, L, n):
        # the opening and closing sequences are always the same
        upper_start = [0] + [1 if i%2==0 else 0 for i in range(n)]
        lower_start = [0] + [0]*n
        upper_finish = [0 if i%2==0 else 1 for i in range(n-1)] # upper is here defined as the last arm to be excited (does not necessarily correspond to the real upper arm)
        lower_finish = [0]*(n-1)

        # the diamonds consist of two types of sequences which alternate
        g_to_e = [0]*(n-1) + [1 if i%2==0 else 0 for i in range(n)]
        e_to_g = [0 if i%2==0 else 1 for i in range(n-1)] + [0]*n

        # build the sequences
        upper_path = upper_start
        lower_path = lower_start

        for loop in range(L):
            if loop%2==0:
                upper_path += e_to_g
                lower_path += g_to_e
            else:
                upper_path += g_to_e
                lower_path += e_to_g

        # now, append the finish sequence depending on which arm was last excited
        if L%2==1:
            lower_path += upper_finish
            upper_path += lower_finish
        else:
            lower_path += lower_finish
            upper_path += upper_finish

        # create strings
        lower_path_str = "".join([str(s) for s in lower_path])
        upper_path_str = "".join([str(s) for s in upper_path])

        return [lower_path_str+"0", lower_path_str+"1", upper_path_str+"0", upper_path_str+"1"]

    def _write_wavefront_params(self, N):
        """Emit the per-pulse wavefront keys: Zernike coefficients, the sampled
        beam file, and tip/tilt.

        All three are optional. Zernike coefficients are given as
        ``pulse_params['zernike_coeffs'] = {noll_index: value}`` and are written
        for every pulse; a sampled beam is selected with ``wtype = 'interpolated'``
        and needs ``pulse_params['beam_file']``.

        Note that ais++ applies Zernike aberrations to the *phase* but not to the
        wavefront gradient, and applies neither to interpolated beams -- see
        KNOWN_ISSUES.md in the ais++ repository.
        """
        wtype = self.pulse_params['wtype']

        # --- Zernike coefficients, uniform across the sequence ---
        zernike_coeffs = self.pulse_params.get('zernike_coeffs') or {}
        if zernike_coeffs and wtype == 'interpolated':
            raise ValueError(
                "zernike_coeffs cannot be combined with wtype='interpolated': the "
                "sampled grid is the complete description of the beam, so ais++ "
                "ignores the coefficients and the result would silently omit them."
            )
        for noll_index, value in sorted(zernike_coeffs.items()):
            self.aisi_file.write(
                f"zernikecoeff_{noll_index} " + " ".join(str(value) for _ in range(N)) + " \n"
            )

        # --- sampled beam file ---
        beam_file = self.pulse_params.get('beam_file')
        if wtype == 'interpolated':
            if not beam_file:
                raise ValueError(
                    "wtype='interpolated' requires pulse_params['beam_file'], the "
                    "HDF5 grid written by aisoptics' AISPPExporter."
                )
            self.aisi_file.write(
                "beaminterpolationparamsfilenames " + " ".join(str(beam_file) for _ in range(N)) + " \n"
            )
        elif beam_file:
            raise ValueError(
                f"pulse_params['beam_file'] is set but wtype is {wtype!r}; the file "
                f"would be ignored. Set wtype='interpolated' to use it."
            )

        # --- tip/tilt, degrees ---
        tiptiltx = self.pulse_params.get('tiptiltx', 0.0)
        tiptilty = self.pulse_params.get('tiptilty', 0.0)
        if tiptiltx or tiptilty:
            if wtype != 'interpolated':
                raise ValueError(
                    f"tiptiltx/tiptilty are only applied to wtype='interpolated' "
                    f"beams, but wtype is {wtype!r}; they would be ignored."
                )
            self.aisi_file.write("tiptiltx " + " ".join(str(tiptiltx) for _ in range(N)) + " \n")
            self.aisi_file.write("tiptilty " + " ".join(str(tiptilty) for _ in range(N)) + " \n")

    def _make_palindrome(self, base, include_center_twice=False):
        # base = [T1, ..., TL]
        if include_center_twice:
            # even total diamonds: [T1,...,TL, TL,...,T1]
            return base + base[::-1]
        else:
            # odd total diamonds: [T1,...,TL-1, TL, TL-1,...,T1]
            return base + base[-2::-1]

    def _write_ultranarrow_MZ_Lloops(self):
        """
        Write a symmetric L-loop ultranarrow MZ sequence (L diamonds).
        T_base: list of interrogation times [T1, ..., TL].
                The full symmetric list is constructed as [T1,...,TL,...,T1].
        Keeps single-loop behavior if len(T_base) == 1.
        """

        # ---- unpack
        T_base = self.sequence_params['interrogation_time']
        L = self.sequence_params['loopnumber']
        v0 = self.cloud_params['v0'][2]
        n  = self.sequence_params['lmt_order']       # must be odd
        assert n % 2 == 1, "ultranarrow requires odd lmt_order"
        nlmt = n - 1                                  # π-kicks per LMT block

        dt_lmt    = self.sequence_params['dt_lmt']
        rabi_freq = self.pulse_params['rabi_freq']
        wtype     = self.pulse_params['wtype']
        phi0      = self.pulse_params['phi0']
        kx_psr    = self.pulse_params['kx_psr']
        ky_psr    = self.pulse_params['ky_psr']
        beam_radius = self.pulse_params['beam_radius']
        baseline    = self.pulse_params['baseline']

        # timing constants (match your single-loop conventions)
        dt_bs = pi / (2 * rabi_freq)
        dt_pi = pi / (    rabi_freq)

        # build symmetric interrogation list
        ndiamonds=L
        L = len(T_base)
        assert L >= 1

        # 'loopnumber' and len('interrogation_time') are two different things and
        # both drive the geometry: the palindrome length below sets the number of
        # diamonds actually written, while the sign and e->g schedules further
        # down are rebuilt from loopnumber directly. If they disagree the pulse
        # schedule and the path list describe different interferometers, and
        # nothing complains -- you just get no interference. The palindrome makes
        # D = 2*len(T_base) for even loopnumber and 2*len(T_base)-1 for odd, so:
        expected_base = (ndiamonds + 1) // 2
        assert L == expected_base, (
            f"loopnumber={ndiamonds} needs len(interrogation_time)={expected_base}, "
            f"got {L}. The full interrogation list is built as a palindrome, so "
            f"only the first ceil(loopnumber/2) times are given explicitly."
        )

        if ndiamonds==1:
            T_full = [T_base[0]]        # single diamond
        else:
            if ndiamonds%2==0:
                T_full = self._make_palindrome(T_base, include_center_twice=True)
            else:
                T_full = self._make_palindrome(T_base, include_center_twice=False)
            #T_full = list(T_base) + list(T_base[-2::-1])  # [T1,...,TL-1, TL, TL-1,...,T1]
        D = len(T_full)                  # number of diamonds in the full chain
        assert D == ndiamonds, (
            f"internal: built {D} diamonds for loopnumber={ndiamonds}"
        )

        # The LMT blocks leave an uncorrected vertical arm separation of order
        # (hbar k/m)*dt_pi*n(n-1)/2. This is a property of the pulse sequence, not
        # of the frame -- it is identical at Omega = 0 -- and it scales with the
        # pi-pulse duration. At n = 101 and a 10 kHz Rabi frequency it reaches
        # 3.3 mm, which is larger than a typical coherence length, so nothing
        # interferes at all and the run silently returns no fringe. 100 kHz brings
        # the same case to 0.34 mm. Warn rather than abort: the estimate is an
        # order-of-magnitude bound and a caller may legitimately be exploring.
        residual_sep = float(hbar * kz / m) * float(dt_pi) * n * (n - 1) / 2
        coherence_length = self.simulation_params.get('coherencelength')
        if coherence_length is not None and residual_sep > float(coherence_length):
            import warnings
            warnings.warn(
                f"LMT residual arm separation ~{residual_sep*1e3:.2f} mm exceeds "
                f"coherencelength {float(coherence_length)*1e3:.2f} mm at "
                f"lmt_order={n}, rabi_freq={float(rabi_freq):.3g} rad/s. The arms "
                f"will not interfere. Shorten the pi pulse (raise rabi_freq) or "
                f"raise coherencelength.",
                RuntimeWarning,
                stacklevel=2,
            )

        # ----- compute the effective interrogation times
        for i in range(len(T_full)):
            T_full[i] = T_full[i] - 2*(n-1)*(dt_lmt + dt_pi)
            assert T_full[i] > 0, "Interrogation time too short for LMT blocks"

        # ----- pulse arrays
        start_times, end_times = [], []
        kz_vals, omega_vals    = [], []

        # compute the signs
        lmt_order = self.sequence_params["lmt_order"]
        loopnumber = self.sequence_params["loopnumber"]
        signs = [1 if i%2==0 else -1 for i in range(lmt_order)]
        signs += [1 if i%2==0 else -1 for i in range(2*lmt_order-1)] * loopnumber
        signs += [1 if i%2==0 else -1 for i in range(lmt_order)]

        # compute the e->g flags
        is_e_to_g = [False if i%2==0 else True for i in range(lmt_order)]
        for loop in range(loopnumber):
            is_e_to_g += [True if i%2==0 else False for i in range(lmt_order-1)]
            is_e_to_g += [False]
            is_e_to_g += [True if i%2==0 else False for i in range(lmt_order-1)]
        is_e_to_g += [True if i%2==0 else False for i in range(lmt_order-1)]
        is_e_to_g += [False]

        # ---- helpers (local) ----

        def _emit_mirror_block_times(t0_start, lmt_order):
            t0s = [t0_start + k * (dt_pi + dt_lmt) for k in range(2*lmt_order-1)]
            t1s = [t + dt_pi for t in t0s]
            start_times.extend(t0s); end_times.extend(t1s)

        def _emit_init_block_times(t0_start, lmt_order):
            t0s = [t0_start + k * (dt_pi + dt_lmt) for k in range(lmt_order-1)]
            t1s = [t + dt_pi for t in t0s]
            start_times.extend(t0s); end_times.extend(t1s)

        def _emit_final_block_times(t0_start, lmt_order):
            _emit_init_block_times(t0_start, lmt_order)

        def _mirror_block(t0_start, vz0):
            # use_global_k_for_up=True reproduces your block-2: subtract ħ*kz/m for s=+1
            _emit_mirror_block_times(t0_start, lmt_order)
            local_kz, local_om = [], []

            # compute the signs and e2g tags
            signs = [1 if i%2==0 else -1 for i in range(2*lmt_order-1)]
            e2g   = [True if i%2==0 else False for i in range(lmt_order-1)]
            e2g  += [True]
            e2g  += [True if i%2==0 else False for i in range(lmt_order-1)]
            # debug prints
            #print("Mirror")
            #print(signs)
            #print(e2g)
            recoil_sum = lmt_order*hbar*kz/m
            for k in range(2*lmt_order-1):
                t_i   = k * (dt_pi + dt_lmt)
                v_t   = vz0 - g * t_i # v_COM
                s     = signs[k]
                is_e  = e2g[k]

                v_tot = v_t + recoil_sum
                
                if k <= lmt_order-2:
                    # adressing upper arm, being decelerated
                    recoil_sum -= hbar*kz/m
                elif k == lmt_order-1:
                    recoil_sum = hbar*kz/m
                else:
                    recoil_sum += hbar*kz/m

                if s == +1:
                    om = detuning(v_tot, is_e); kz_i = om / c
                    local_kz.append(kz);  local_om.append(om)
                else:
                    om = detuning(-v_tot, is_e); kz_i = om / c
                    local_kz.append(-kz); local_om.append(om)

            return local_kz, local_om
        
        def _init_block(t0_start, vz0):
            # after the pi/2 pulse
            _emit_init_block_times(t0_start, lmt_order)
            local_kz, local_om = [], []

            signs = [-1 if i%2==0 else 1 for i in range(lmt_order-1)]
            e2g   = [True if i%2==0 else False for i in range(lmt_order-1)]
            # debug prints
            #print("init")
            #print(signs)
            #print(e2g)

            recoil_sum = hbar*kz/m
            for k in range(lmt_order-1):
                t_i   = k * (dt_pi + dt_lmt)
                v_t   = vz0 - g * t_i # v_COM
                s     = signs[k]
                is_e  = e2g[k]

                v_tot = v_t + recoil_sum

                recoil_sum += hbar*kz/m

                if s == +1:
                    om = detuning(v_tot, is_e); kz_i = om / c
                    local_kz.append(kz);  local_om.append(om)
                else:
                    om = detuning(-v_tot, is_e); kz_i = om / c
                    local_kz.append(-kz); local_om.append(om)
            return local_kz, local_om
        
        def _final_block(t0_start, vz0):
            # after the pi/2 pulse
            _emit_final_block_times(t0_start, lmt_order)
            local_kz, local_om = [], []

            signs = [1 if i%2==0 else -1 for i in range(lmt_order-1)]
            e2g   = [True if i%2==0 else False for i in range(lmt_order-1)]
            # debug prints
            #print("final")
            #print(signs)
            #print(e2g)

            recoil_sum = lmt_order*hbar*kz/m 
            for k in range(lmt_order-1):
                t_i   = k * (dt_pi + dt_lmt)
                v_t   = vz0 - g * t_i # v_COM
                s     = signs[k]
                is_e  = e2g[k]

                v_tot = v_t + recoil_sum
                # debug prints
                #print("vtot = ", v_tot)

                recoil_sum -= hbar*kz/m

                if s == +1:
                    om = detuning(v_tot, is_e); kz_i = om / c
                    local_kz.append(kz);  local_om.append(om)
                else:
                    om = detuning(-v_tot, is_e); kz_i = om / c
                    local_kz.append(-kz); local_om.append(om)
            return local_kz, local_om

        # ---- time cursor and beam-splitters
        t = self.sequence_params['t_init']
        # initial π/2
        start_times.append(t); end_times.append(t + dt_bs)
        # detuning for initial π/2 (as before)
        om = detuning(v0, False); kz_i = om / c
        kz_vals.append(kz_i); omega_vals.append(om)
        t += dt_bs + dt_lmt

        if lmt_order != 1:
            # initial acceleration block
            kz_init, om_init = _init_block(t, v0 - g*t)
            kz_vals += kz_init
            omega_vals += om_init

            t = end_times[-1] 

        # Position-closure correction: the init and final LMT blocks each accumulate
        # Δz_init = ħk/m·dt·n(n-1)/2 of arm separation (both with the same sign).
        # To close the interferometer, shorten the dead-times adjacent to those blocks
        # by δ_half = (n-1)·(dt_pi+dt_lmt)/2 each, so the arms converge by 2·Δz_init
        # before the final block begins.
        #
        # Both ends are shortened, for every D.  This used to carry a (-1)**D
        # factor on the last dead time, which LENGTHENED it for odd D and left a
        # residual arm mismatch of exactly
        #     2·δ_half·n·v_rec = n(n-1)·(dt_pi+dt_lmt)·ħk/m,
        # because the arms are separating at n·v_rec across that interval.  The
        # error grows as n²: 6.7e-3 m at n=101 and 4.1e-2 m at n=251, against a
        # 1 mm interference tolerance, so every odd-D LMT sequence failed to
        # close while every even-D one was fine.  Measured over D=1..8 with
        # ais++, removing the factor takes odd D from 6.65e-3 m to 6.57e-6 m --
        # which is v_rec × the 1 ms detection delay, i.e. the two output ports
        # drifting apart by one photon recoil, the floor rather than an error.
        # Even D is unchanged at 7.23e-5 m.
        D = len(T_full)
        delta_half = (lmt_order - 1) * (dt_pi + dt_lmt) / 2

        # Even D needs one further correction.  Each mirror block swaps which arm
        # is ahead, so after D of them an even-D sequence ends with the arms in
        # their original roles, and the init/final block displacements add rather
        # than cancel — leaving 2·δ_half·v_rec of separation.  The arms close at
        # n·v_rec across the final dead time, so removing it costs 2·δ_half/n of
        # that interval.  Odd D already lands on the floor and needs nothing.
        # Measured over D=1..6, n=11/101/251: this puts every case at 1.00–1.02×
        # the v_rec·t_detect floor, taking even D from 7.23e-5 m to 6.64e-6 m.
        extra_last = (2 * delta_half / lmt_order) if D % 2 == 0 else 0.0

        # walk diamonds
        for idx, Ti in enumerate(T_full):
            # first dead time — shortened for first diamond
            t += Ti - delta_half if idx == 0 else Ti
            vz0 = v0-g*t
            # mirror block
            kz_mirror, om_mirror = _mirror_block(t, vz0)
            kz_vals += kz_mirror
            omega_vals += om_mirror
            # update time
            t = end_times[-1]
            # second dead time — shortened for the last diamond, regardless of parity
            if idx == D - 1:
                t += Ti - delta_half - extra_last
            else:
                t += Ti

        if lmt_order != 1:
            # final deceleration block
            kz_final, om_final = _final_block(t, v0 - g*t)
            kz_vals += kz_final
            omega_vals += om_final

            t = end_times[-1] + dt_lmt

        # final π/2
        start_times.append(t); end_times.append(t + dt_bs)
        # final BS detuning (as before, no recoil addition)
        v_t = v0 - g * t
        om = detuning(v_t, False); kz_i = om / c
        kz_vals.append(kz_i); omega_vals.append(om)

        # ---- transverse components and file emission (sizes derived automatically)
        N = len(start_times)  # total pulses
        kx = np.zeros(N); ky = np.zeros(N)
        kx[-1] = kx_psr; ky[-1] = ky_psr

        self.aisi_file.write("# Pulse parameters\n")
        self.aisi_file.write("t0 " + " ".join(str(x) for x in start_times) + " \n")
        self.aisi_file.write("t1 " + " ".join(str(x) for x in end_times) + " \n")
        self.aisi_file.write("kx " + " ".join(str(x) for x in kx) + " \n")
        self.aisi_file.write("ky " + " ".join(str(y) for y in ky) + " \n")
        self.aisi_file.write("kz " + " ".join(str(k) for k in kz_vals) + " \n")
        self.aisi_file.write("omega " + " ".join(str(w) for w in omega_vals) + " \n")
        self.aisi_file.write("rabifreq " + " ".join(str(rabi_freq/(2*pi)) for _ in range(N)) + " \n")
        self.aisi_file.write("wtype " + " ".join(wtype for _ in range(N)) + " \n")
        # phase on the very last pulse (keep your convention)
        self.aisi_file.write("phi0 " + " ".join(str(phi0) if i == (N-1) else "0" for i in range(N)) + " \n")
        # zero chirps here
        self.aisi_file.write("kxchirp " + " ".join("0" for _ in range(N)) + " \n")
        self.aisi_file.write("kychirp " + " ".join("0" for _ in range(N)) + " \n")
        self.aisi_file.write("kzchirp " + " ".join("0" for _ in range(N)) + " \n")
        self.aisi_file.write("frequencychirp " + " ".join("0" for _ in range(N)) + "\n")
        self.aisi_file.write("waist " + " ".join(str(self.pulse_params['waist']) for _ in range(N)) + " \n")
        self.aisi_file.write("focallength " + " ".join(str(self.pulse_params['focallength']) for _ in range(N)) + " \n")
        # z-laser per sign (we can infer from kz sign we emitted):
        self.aisi_file.write("zlaser " + " ".join(["0"]*(N)) + "\n")
        self.aisi_file.write("beamradius " + " ".join(str(beam_radius) for _ in range(N)) + " \n")
        self.aisi_file.write("baseline " + " ".join(str(baseline) for _ in range(N)) + " \n")

        self._write_wavefront_params(N)

    def _write_chirped_sequence_ultranarrow_MZ(self):
        # get the initial vertical velocity
        v0 = self.cloud_params['v0'][2]

        # get the lmt order
        n = self.sequence_params['lmt_order']

        # get the duration of the lmt pulses
        dt_bs = pi / (2 * self.pulse_params['rabi_freq'])
        dt_lmt = self.sequence_params['dt_lmt']
        pulse_duration = np.pi / (self.pulse_params['rabi_freq'])

        dt_acc = dt_lmt + pulse_duration

        if n == 1:
            dt_acc = np.inf # sets the chirp to 0 for the first block

        # compute the detunined frequencies base on the initial velocity (one for each of the for blocks)
        omega_chirp = self.sequence_params['frequencychirp'] * (kz*g - hbar*kz**2/(m*(dt_acc)))
        kchirp = 0 # assume 0 kchirp for now
        # bs
        omega0_detuned_plus_pi_over_two = detuning(v0 + omega_chirp*dt_bs/kz)
        omega0_detuned_minus_pi_over_two = detuning(-v0 - omega_chirp*dt_bs/kz)
        
        kz_detuned_plus_pi_over_two = omega0_detuned_plus_pi_over_two / c
        kz_detuned_minus_pi_over_two = omega0_detuned_minus_pi_over_two / c

        # block 1
        omega0_detuned_plus_b1 = detuning(v0 + hbar*kz/m + omega_chirp*pulse_duration/kz)
        omega0_detuned_minus_b1 = detuning(-v0 - hbar*kz/m - omega_chirp*pulse_duration/kz)

        kz_detuned_plus_b1 = omega0_detuned_plus_b1 / c
        kz_detuned_minus_b1 = omega0_detuned_minus_b1 / c

        # block 2
        t0_b2 = self.sequence_params['t_init'] + self.sequence_params['interrogation_time'] - (n-1)/2*(pulse_duration + dt_lmt)
        omega0_detuned_plus_b2 = detuning(v0 + n*hbar*kz/m - hbar*kz/(m*dt_acc)*t0_b2)
        omega0_detuned_minus_b2 = detuning(-v0 - n*hbar*kz/m + hbar*kz/(m*dt_acc)*t0_b2)

        kz_detuned_plus_b2 = omega0_detuned_plus_b2 / c
        kz_detuned_minus_b2 = omega0_detuned_minus_b2 / c

        # block 3
        t0_b3 = self.sequence_params['t_init'] + self.sequence_params['interrogation_time']
        omega0_detuned_plus_b3 = detuning(v0 - hbar*kz/(m*dt_acc)*t0_b3)
        omega0_detuned_minus_b3 = detuning(-v0 + hbar*kz/(m*dt_acc)*t0_b3)

        kz_detuned_plus_b3 = omega0_detuned_plus_b3 / c
        kz_detuned_minus_b3 = omega0_detuned_minus_b3 / c

        # block 4
        t0_b4 = self.sequence_params['t_init'] + 2*self.sequence_params['interrogation_time'] - (n-1)/2*(pulse_duration + dt_lmt)
        omega0_detuned_plus_b4 = detuning(v0 + n*hbar*kz/m - hbar*kz/(m*dt_acc)*t0_b4)
        omega0_detuned_minus_b4 = detuning(-v0 - n*hbar*kz/m + hbar*kz/(m*dt_acc)*t0_b4)

        kz_detuned_plus_b4 = omega0_detuned_plus_b4 / c
        kz_detuned_minus_b4 = omega0_detuned_minus_b4 / c

        # compute times etc as before
        z0 = self.cloud_params['x0'][2]
        t_init = self.sequence_params['t_init']
        lmt_order = self.sequence_params['lmt_order']
        dt_lmt = self.sequence_params['dt_lmt']
        T = self.sequence_params['interrogation_time']
        rabi_freq = self.pulse_params['rabi_freq']
        wtype = self.pulse_params['wtype']
        phi0 = self.pulse_params['phi0']
        kx_psr = self.pulse_params['kx_psr']
        ky_psr = self.pulse_params['ky_psr']
        beam_radius = self.pulse_params['beam_radius']
        baseline = self.pulse_params['baseline']

        # Zernike polynomials params
        # format: {noll_index: [coeff1_up, coeff2_up, coeff3_up, coeff1_down, coeff2_down, coeff3_down]}
        zernike_params = self.pulse_params['zernike_params']

        assert(dt_lmt > 0)
        
        dt1 = dt_lmt + np.pi/(2*rabi_freq) # makes the spacing same as for the pi pulses
        dt2 = dt_lmt
        dt3 = dt_lmt
        dt4 = dt_lmt + np.pi/(2*rabi_freq)

        assert(lmt_order%2 == 1)
        nlmt = int((lmt_order-1))

        # calculate the time sequence
        dt_bs = pi/(2*rabi_freq)
        dt_pi = pi/(rabi_freq)

        # compute the effective interrogation time
        T = T - 2*nlmt*(dt_lmt + dt_pi)

        lmt_pulse_index = np.arange(0, nlmt+1)
        t_start_shifted = lmt_pulse_index * (dt_pi + dt_lmt)
        t_end_shifted   = t_start_shifted + dt_pi
        if nlmt > 1:
            t_tot           = t_end_shifted[-1]
        else:
            t_tot = mp.mpf('0')
        
        t_bs1 = t_init
        t0 = dt_bs + t_init + dt1
        t1 = t0 + t_tot + T
        t_pi = t1 + t_tot + dt2
        t2 = t_pi + dt_pi + dt3
        t3 = t2 + t_tot + T
        t_bs2 = t3 + t_tot + dt4

        start_times = []
        end_times = []

        # initial pi/2 pulse
        start_times.append(t_bs1)
        end_times.append(t_bs1 + dt_bs)

        # LMT block 1
        for i in range(nlmt):
            start_times.append(t0 + t_start_shifted[i])
            end_times.append(t0 + t_end_shifted[i])

        # LMT block 2
        for i in range(nlmt):
            start_times.append(t1 + t_start_shifted[i])
            end_times.append(t1 + t_end_shifted[i])
        
        # pi pulse upper
        start_times.append(t_pi)
        end_times.append(t_pi + dt_pi)
        # pi pulse lower
        start_times.append(t_pi + dt_pi + dt_lmt)
        end_times.append(t_pi + 2*dt_pi + dt_lmt)

        # LMT block 3
        for i in range(nlmt):
            start_times.append(t2 + t_start_shifted[i])
            end_times.append(t2 + t_end_shifted[i])

        # LMT block 4
        for i in range(nlmt):
            start_times.append(t3 + t_start_shifted[i])
            end_times.append(t3 + t_end_shifted[i])

        # final pi/2 pulse
        start_times.append(t_bs2)
        end_times.append(t_bs2 + dt_bs)

        # compute the direction of the pulses
        sign = []
        for i in range(0,nlmt+1):
            sign.append((-1)**i)
        for i in range(0,nlmt+1): # block 2 and first mirror pulse (upper)
            sign.append((-1)**(nlmt+i))
        sign.append(1) # mirror pulse (lower)
        for i in range(0,nlmt+1):
            sign.append((-1)**(3*nlmt+i))

        # compute the kz and omega values for each block
        kz_detuned_values = []
        omega0_detuned_values = []
        # pi/2 pulse
        kz_detuned_values.append(kz_detuned_plus_pi_over_two)
        omega0_detuned_values.append(omega0_detuned_plus_pi_over_two)
        # first block is index 1 to nlmt
        for i in range(1,nlmt+1):
            if sign[i] == 1:
                kz_detuned_values.append(kz_detuned_plus_b1)
                omega0_detuned_values.append(omega0_detuned_plus_b1)
            else:
                kz_detuned_values.append(-kz_detuned_minus_b1)
                omega0_detuned_values.append(omega0_detuned_minus_b1)

        # second block is index nlmt+1 to 2*nlmt+1 (inlcuding the mirror pulse)
        for i in range(nlmt+1):
            if sign[i+nlmt+1] == 1:
                kz_detuned_values.append(kz_detuned_plus_b2)
                omega0_detuned_values.append(omega0_detuned_plus_b2)
            else:
                kz_detuned_values.append(-kz_detuned_minus_b2)
                omega0_detuned_values.append(omega0_detuned_minus_b2)

        # third block is index 2*nlmt+2 to 3*nlmt+2 (including the second mirror pulse)
        for i in range(nlmt+1):
            if sign[i+2*nlmt+2] == 1:
                kz_detuned_values.append(kz_detuned_plus_b3)
                omega0_detuned_values.append(omega0_detuned_plus_b3)
            else:
                kz_detuned_values.append(-kz_detuned_minus_b3)
                omega0_detuned_values.append(omega0_detuned_minus_b3)

        # fourth block is index 3*nlmt+2 to 4*nlmt+3 (including the beam splitter pulse)
        for i in range(nlmt+1):
            if sign[i+3*nlmt+2] == 1:
                kz_detuned_values.append(kz_detuned_plus_b4)
                omega0_detuned_values.append(omega0_detuned_plus_b4)
            else:
                kz_detuned_values.append(-kz_detuned_minus_b4)
                omega0_detuned_values.append(omega0_detuned_minus_b4)

        # transverse wavevector components
        kx = np.zeros(4+4*nlmt)
        ky = np.zeros(4+4*nlmt)
        kx[-1] = kx_psr #psr
        ky[-1] = ky_psr #psr

        # write the start times
        self.aisi_file.write("# Pulse parameters\n")
        self.aisi_file.write("t0 ")
        for t_ in start_times:
            self.aisi_file.write(str(t_) + " ")
        self.aisi_file.write("\n")
        # write the end times
        self.aisi_file.write("t1 ")
        for t_ in end_times:
            self.aisi_file.write(str(t_) + " ")
        self.aisi_file.write("\n")
        # write the kx values
        self.aisi_file.write("kx ")
        for kx_ in kx:
            self.aisi_file.write(str(kx_) + " ")
        self.aisi_file.write("\n")
        # write the ky values
        self.aisi_file.write("ky ")
        for ky_ in ky:
            self.aisi_file.write(str(ky_) + " ")
        self.aisi_file.write("\n")

        # write the kz values for each block
        self.aisi_file.write("kz ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(str(kz_detuned_values[i]) + " ")
        self.aisi_file.write("\n")
        # write the detuned frequencies
        self.aisi_file.write("omega ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(str(omega0_detuned_values[i]) + " ")
        self.aisi_file.write("\n")
        # write the rabi frequency
        self.aisi_file.write("rabifreq ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(str(rabi_freq/(2*pi)) + " ")
        self.aisi_file.write("\n")
        # write wavefront type
        self.aisi_file.write("wtype ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(wtype + " ")
        self.aisi_file.write("\n")
        # write phi0
        self.aisi_file.write("phi0 ")
        for i in range(4+4*nlmt):
            if i == 4+4*nlmt-1:
                self.aisi_file.write(str(phi0) + "\n")
            else:
                self.aisi_file.write("0 ")
        # write the kchirp and frequency chirp (0 for this case)
        self.aisi_file.write("kxchirp ")
        for i in range(4+4*nlmt):
            self.aisi_file.write("0 ")
        self.aisi_file.write("\n")
        self.aisi_file.write("kychirp ")
        for i in range(4+4*nlmt):
            self.aisi_file.write("0 ")
        self.aisi_file.write("\n")
        self.aisi_file.write("kzchirp ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(str(-sign[i]*kchirp) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("frequencychirp ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(str(-sign[i]*omega_chirp) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("waist ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(str(self.pulse_params['waist']) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("focallength ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(str(self.pulse_params['focallength']) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("zlaser ")
        for i in range(4+4*nlmt):
            if sign[i] == 1:
                self.aisi_file.write(str(self.pulse_params['zupwardlaser']) + " ")
            else:
                self.aisi_file.write(str(self.pulse_params['zdownwardlaser']) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("beamradius ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(str(beam_radius) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("baseline ")
        for i in range(4+4*nlmt):
            self.aisi_file.write(str(baseline) + " ")
        self.aisi_file.write("\n")

        # Zernike coefficients
        for zernike_noll_index in zernike_params.keys():
            self.aisi_file.write("zernikecoeff_{} ".format(zernike_noll_index))
            # write the coeffs for BS block (first 1 + (n-1)/2 pulses)
            for i in range(0,int(1 + (lmt_order-1)/2)):
                if sign[i] == 1:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][0]) + " ")
                else:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][3]) + " ")
            # write the coeffs for LMT block 1 (next n pulses)
            for i in range(lmt_order):
                if sign[i+int(1 + (lmt_order-1)/2)] == 1:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][1]) + " ")
                else:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][4]) + " ")
            # write the coeffs for LMT block 2 (next 1 + (n-1)/2 pulses)
            for i in range(int(1 + (lmt_order-1)/2)):
                if sign[i+int(1 + (lmt_order-1)/2)+lmt_order] == 1:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][2]) + " ")
                else:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][5]) + " ")
            self.aisi_file.write("\n")
        
    def _write_chirped_sequence_MZ(self):
        # get the initial vertical velocity
        v0 = self.cloud_params['v0'][2]
        # get the v0-detuned frequency and wavevector
        omega0_detuned_plus = detuning(v0)
        omega0_detuned_minus = detuning(-v0)
        kz_detuned_plus = omega0_detuned_plus / c
        kz_detuned_minus = omega0_detuned_minus / c
        # compute the absolute values of the frequency chirp and the k-vector chirp
        omega_chirp = self.sequence_params['frequencychirp'] * kz * g
        k_chirp = self.sequence_params['kchirp'] * kz * g / c

        # compute times etc as before
        z0 = self.cloud_params['x0'][2]
        t_init = self.sequence_params['t_init']
        lmt_order = self.sequence_params['lmt_order']
        dt_lmt = self.sequence_params['dt_lmt']
        T = self.sequence_params['interrogation_time']
        rabi_freq = self.pulse_params['rabi_freq']
        wtype = self.pulse_params['wtype']
        phi0 = self.pulse_params['phi0']
        kx_psr = self.pulse_params['kx_psr']
        ky_psr = self.pulse_params['ky_psr']
        beam_radius = self.pulse_params['beam_radius']
        baseline = self.pulse_params['baseline']

        # Zernike polynomials params
        # format: {noll_index: [coeff1_up, coeff2_up, coeff3_up, coeff1_down, coeff2_down, coeff3_down]}
        zernike_params = self.pulse_params['zernike_params']

        if lmt_order == 1:
            assert(dt_lmt == 0)
        else:
            assert(dt_lmt > 0)
        
        dt1 =dt_lmt
        dt2 = dt_lmt
        dt3 = dt_lmt
        dt4 = dt_lmt

        assert(lmt_order%2 == 1)
        nlmt = int((lmt_order-1)/2)

        # calculate the time sequence
        dt_bs = pi/(2*rabi_freq)
        dt_pi = pi/(rabi_freq)

        lmt_pulse_index = np.arange(0, nlmt+1)
        t_start_shifted = lmt_pulse_index * (dt_pi + dt_lmt)
        t_end_shifted   = t_start_shifted + dt_pi
        if nlmt > 1:
            t_tot           = t_end_shifted[-1]
        else:
            t_tot = mp.mpf('0')
        
        t_bs1 = t_init
        t0 = dt_bs + t_init + dt1
        t1 = t0 + t_tot + T
        t_pi = t1 + t_tot + dt2
        t2 = t_pi + dt_pi + dt3
        t3 = t2 + t_tot + T
        t_bs2 = t3 + t_tot + dt4

        start_times = []
        end_times = []

        # initial pi/2 pulse
        start_times.append(t_bs1)
        end_times.append(t_bs1 + dt_bs)

        # LMT block 1
        for i in range(nlmt):
            start_times.append(t0 + t_start_shifted[i])
            end_times.append(t0 + t_end_shifted[i])

        # LMT block 2
        for i in range(nlmt):
            start_times.append(t1 + t_start_shifted[i])
            end_times.append(t1 + t_end_shifted[i])
        
        # pi pulse
        start_times.append(t_pi)
        end_times.append(t_pi + dt_pi)

        # LMT block 3
        for i in range(nlmt):
            start_times.append(t2 + t_start_shifted[i])
            end_times.append(t2 + t_end_shifted[i])

        # LMT block 4
        for i in range(nlmt):
            start_times.append(t3 + t_start_shifted[i])
            end_times.append(t3 + t_end_shifted[i])

        # final pi/2 pulse
        start_times.append(t_bs2)
        end_times.append(t_bs2 + dt_bs)

        # compute the direction of the pulses
        sign = []
        for i in range(0,nlmt+1):
            sign.append((-1)**i)
        for i in range(0,2*nlmt+1):
            sign.append((-1)**(nlmt+i))
        for i in range(0,nlmt+1):
            sign.append((-1)**(3*nlmt+i))

        kx = np.zeros(3+4*nlmt)
        ky = np.zeros(3+4*nlmt)
        kx[-1] = kx_psr #psr
        ky[-1] = ky_psr #psr

        # write the start times
        self.aisi_file.write("# Pulse parameters\n")
        self.aisi_file.write("t0 ")
        for t_ in start_times:
            self.aisi_file.write(str(t_) + " ")
        self.aisi_file.write("\n")
        # write the end times
        self.aisi_file.write("t1 ")
        for t_ in end_times:
            self.aisi_file.write(str(t_) + " ")
        self.aisi_file.write("\n")
        # write the kx values
        self.aisi_file.write("kx ")
        for kx_ in kx:
            self.aisi_file.write(str(kx_) + " ")
        self.aisi_file.write("\n")
        # write the ky values
        self.aisi_file.write("ky ")
        for ky_ in ky:
            self.aisi_file.write(str(ky_) + " ")
        self.aisi_file.write("\n")
        # write the kz values
        self.aisi_file.write("kz ")
        for i in range(3+4*nlmt):
            if sign[i] == 1:
                self.aisi_file.write(str(kz_detuned_plus) + " ")
            else:
                self.aisi_file.write(str(-kz_detuned_minus) + " ")
        self.aisi_file.write("\n")
        # write the detuned frequencies
        self.aisi_file.write("omega ")
        for i in range(3+4*nlmt):
            if sign[i] == 1:
                self.aisi_file.write(str(omega0_detuned_plus) + " ")
            else:
                self.aisi_file.write(str(omega0_detuned_minus) + " ")
        self.aisi_file.write("\n")
        # write the rabi frequency
        self.aisi_file.write("rabifreq ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(rabi_freq/(2*pi)) + " ")
        self.aisi_file.write("\n")
        # write wavefront type
        self.aisi_file.write("wtype ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(wtype + " ")
        self.aisi_file.write("\n")
        # write phi0
        self.aisi_file.write("phi0 ")
        for i in range(3+4*nlmt):
            if i == 3+4*nlmt-1:
                self.aisi_file.write(str(phi0) + "\n")
            else:
                self.aisi_file.write("0 ")
        # write the kchirp and frequency chirp (0 for this case)
        self.aisi_file.write("kxchirp ")
        for i in range(3+4*nlmt):
            self.aisi_file.write("0 ")
        self.aisi_file.write("\n")
        self.aisi_file.write("kychirp ")
        for i in range(3+4*nlmt):
            self.aisi_file.write("0 ")
        self.aisi_file.write("\n")
        self.aisi_file.write("kzchirp ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(-sign[i]*k_chirp) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("frequencychirp ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(-sign[i]*omega_chirp) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("waist ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(self.pulse_params['waist']) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("focallength ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(self.pulse_params['focallength']) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("zlaser ")
        for i in range(3+4*nlmt):
            if sign[i] == 1:
                self.aisi_file.write(str(self.pulse_params['zupwardlaser']) + " ")
            else:
                self.aisi_file.write(str(self.pulse_params['zdownwardlaser']) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("beamradius ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(beam_radius) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("baseline ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(baseline) + " ")
        self.aisi_file.write("\n")

        # paths to interpolation params files
        self.aisi_file.write("beaminterpolationparamsfilenames ")
        for i in range(3+4*nlmt):
            if sign[i] == 1:
                self.aisi_file.write(self.pulse_params['beaminterpolationparamsfilenames'][0] + " ")
            else:
                self.aisi_file.write(self.pulse_params['beaminterpolationparamsfilenames'][1] + " ")
        self.aisi_file.write("\n")
        

    def _write_auto_stepwise_detuning_MZ(self):
        v0 = self.cloud_params['v0'][2] # z component of velocity
        z0 = self.cloud_params['x0'][2]
        t_init = self.sequence_params['t_init']
        lmt_order = self.sequence_params['lmt_order']
        dt_lmt = self.sequence_params['dt_lmt']
        T = self.sequence_params['interrogation_time']
        rabi_freq = self.pulse_params['rabi_freq']
        wtype = self.pulse_params['wtype']
        phi0 = self.pulse_params['phi0']
        kx_psr = self.pulse_params['kx_psr']
        ky_psr = self.pulse_params['ky_psr']
        beam_radius = self.pulse_params['beam_radius']
        baseline = self.pulse_params['baseline']

        # Zernike polynomials params
        # format: {noll_index: [coeff1, coeff2, coeff3]}
        zernike_params = self.pulse_params['zernike_params']

        if lmt_order == 1:
            assert(dt_lmt == 0)
        else:
            assert(dt_lmt > 0)
        
        dt1 =dt_lmt
        dt2 = dt_lmt
        dt3 = dt_lmt
        dt4 = dt_lmt

        assert(lmt_order%2 == 1)
        nlmt = int((lmt_order-1)/2)

        # calculate the time sequence
        dt_bs = pi/(2*rabi_freq)
        dt_pi = pi/(rabi_freq)

        lmt_pulse_index = np.arange(0, nlmt+1)
        t_start_shifted = lmt_pulse_index * (dt_pi + dt_lmt)
        t_end_shifted   = t_start_shifted + dt_pi
        if nlmt > 1:
            t_tot           = t_end_shifted[-1]
        else:
            t_tot = mp.mpf('0')
        
        t_bs1 = t_init
        t0 = dt_bs + t_init + dt1
        t1 = t0 + t_tot + T
        t_pi = t1 + t_tot + dt2
        t2 = t_pi + dt_pi + dt3
        t3 = t2 + t_tot + T
        t_bs2 = t3 + t_tot + dt4

        start_times = []
        end_times = []

        # initial pi/2 pulse
        start_times.append(t_bs1)
        end_times.append(t_bs1 + dt_bs)

        # LMT block 1
        for i in range(nlmt):
            start_times.append(t0 + t_start_shifted[i])
            end_times.append(t0 + t_end_shifted[i])

        # LMT block 2
        for i in range(nlmt):
            start_times.append(t1 + t_start_shifted[i])
            end_times.append(t1 + t_end_shifted[i])
        
        # pi pulse
        start_times.append(t_pi)
        end_times.append(t_pi + dt_pi)

        # LMT block 3
        for i in range(nlmt):
            start_times.append(t2 + t_start_shifted[i])
            end_times.append(t2 + t_end_shifted[i])

        # LMT block 4
        for i in range(nlmt):
            start_times.append(t3 + t_start_shifted[i])
            end_times.append(t3 + t_end_shifted[i])

        # final pi/2 pulse
        start_times.append(t_bs2)
        end_times.append(t_bs2 + dt_bs)

        # compute the direction of the pulses
        sign = []
        for i in range(0,nlmt+1):
            sign.append((-1)**i)
        for i in range(0,2*nlmt+1):
            sign.append((-1)**(nlmt+i))
        for i in range(0,nlmt+1):
            sign.append((-1)**(3*nlmt+i))

        # compute the velocities relative to the laser source
        v_rel = []
        for i in range(3+4*nlmt):
            if self.sequence_params['automaticdetuning'] == 0:
                v_rel.append(0)
            else:
                v_rel.append(sign[i]*v(start_times[i],v0,z0))

        # compute the detuned frequencies and wavevectors
        detuned_freq = []
        detuned_kz = []
        for i in range(3+4*nlmt):
            detuned_freq.append(detuning(v_rel[i]))
            detuned_kz.append(sign[i]*detuned_freq[i] / c)

        kx = np.zeros_like(detuned_kz)
        ky = np.zeros_like(detuned_kz)
        kx[-1] = kx_psr #psr
        ky[-1] = ky_psr #psr

        # write the start times
        self.aisi_file.write("# Pulse parameters\n")
        self.aisi_file.write("t0 ")
        for t_ in start_times:
            self.aisi_file.write(str(t_) + " ")
        self.aisi_file.write("\n")
        # write the end times
        self.aisi_file.write("t1 ")
        for t_ in end_times:
            self.aisi_file.write(str(t_) + " ")
        self.aisi_file.write("\n")
        # write the kx values
        self.aisi_file.write("kx ")
        for kx_ in kx:
            self.aisi_file.write(str(kx_) + " ")
        self.aisi_file.write("\n")
        # write the ky values
        self.aisi_file.write("ky ")
        for ky_ in ky:
            self.aisi_file.write(str(ky_) + " ")
        self.aisi_file.write("\n")
        # write the kz values
        self.aisi_file.write("kz ")
        for kz_ in detuned_kz:
            self.aisi_file.write(str(kz_) + " ")
        self.aisi_file.write("\n")
        # write the detuned frequencies
        self.aisi_file.write("omega ")
        for freq_ in detuned_freq:
            self.aisi_file.write(str(freq_) + " ")
        self.aisi_file.write("\n")
        # write the rabi frequency
        self.aisi_file.write("rabifreq ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(rabi_freq/(2*pi)) + " ")
        self.aisi_file.write("\n")
        # write wavefront type
        self.aisi_file.write("wtype ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(wtype + " ")
        self.aisi_file.write("\n")
        # write phi0
        self.aisi_file.write("phi0 ")
        for i in range(3+4*nlmt):
            if i == 3+4*nlmt-1:
                self.aisi_file.write(str(phi0) + "\n")
            else:
                self.aisi_file.write("0 ")
        # write the kchirp and frequency chirp (0 for this case)
        self.aisi_file.write("kxchirp ")
        for i in range(3+4*nlmt):
            self.aisi_file.write("0 ")
        self.aisi_file.write("\n")
        self.aisi_file.write("kychirp ")
        for i in range(3+4*nlmt):
            self.aisi_file.write("0 ")
        self.aisi_file.write("\n")
        self.aisi_file.write("kzchirp ")
        for i in range(3+4*nlmt):
            self.aisi_file.write("0 ")
        self.aisi_file.write("\n")
        self.aisi_file.write("omegachirp ")
        for i in range(3+4*nlmt):
            self.aisi_file.write("0 ")
        self.aisi_file.write("\n")
        self.aisi_file.write("waist ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(self.pulse_params['waist']) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("focallength ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(self.pulse_params['focallength']) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("zlaser ")
        for i in range(3+4*nlmt):
            if sign[i] == 1:
                self.aisi_file.write(str(self.pulse_params['zupwardlaser']) + " ")
            else:
                self.aisi_file.write(str(self.pulse_params['zdownwardlaser']) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("beamradius ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(beam_radius) + " ")
        self.aisi_file.write("\n")
        self.aisi_file.write("baseline ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(baseline) + " ")
        self.aisi_file.write("\n")

        # Zernike coefficients
        for zernike_noll_index in zernike_params.keys():
            self.aisi_file.write("zernikecoeff_{} ".format(zernike_noll_index))
            # write the coeffs for BS block (first 1 + (n-1)/2 pulses)
            for i in range(0,int(1 + (lmt_order-1)/2)):
                if sign[i] == 1:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][0]) + " ")
                else:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][3]) + " ")
            # write the coeffs for LMT block 1 (next n pulses)
            for i in range(lmt_order):
                if sign[i+int(1 + (lmt_order-1)/2)] == 1:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][1]) + " ")
                else:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][4]) + " ")
            # write the coeffs for LMT block 2 (next 1 + (n-1)/2 pulses)
            for i in range(int(1 + (lmt_order-1)/2)):
                if sign[i+int(1 + (lmt_order-1)/2)+lmt_order] == 1:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][2]) + " ")
                else:
                    self.aisi_file.write(str(zernike_params[zernike_noll_index][5]) + " ")


        
