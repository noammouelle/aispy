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
        # write the header
        self._write_header()
        # write the cloud parameters
        self._write_cloud_params()
        # write the potential parameters
        self._write_potential_params()
        # write the simulation parameters
        self._write_simulation_params()
        # write the sequence parameters
        self._write_sequence_params()
        # write IO parameters
        self._write_io_params()
        # write the pulse parameters
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
        self.aisi_file.write('utype {}\n\n'.format(self.potential_params['utype']))

    def _write_simulation_params(self):
        self.aisi_file.write('# Simulation parameters\n')
        self.aisi_file.write('amplitudethreshold {}\n'.format(self.simulation_params['amplitudethreshold']))
        self.aisi_file.write('coherencelength {}\n'.format(self.simulation_params['coherencelength']))
        self.aisi_file.write('usemcbranching {}\n'.format(self.simulation_params['usemcbranching']))
        self.aisi_file.write('usepathselection {}\n'.format(self.simulation_params['usepathselection']))
        self.aisi_file.write('usestaticapprox {}\n'.format(self.simulation_params['usestaticapprox']))
        self.aisi_file.write('ultrafast {}\n'.format(self.simulation_params['ultrafast']))


        # compute the 4 path strings for the 4 main interferometer paths (assuming initially in the ground state)
        if self.sequence_params['ultranarrow'] == 1:
            path0 = "0"*(2*self.sequence_params['lmt_order']-1) + "01"*self.sequence_params['lmt_order']
            path1 = "01"*self.sequence_params['lmt_order'] + "0"*(2*self.sequence_params['lmt_order']-1)
            pathstosimulate = [path0+ "0", path0 + "1",
                               path1 + "0", path1 + "1"]
        else:
            pathstosimulate = ["0" + "10"*self.sequence_params['lmt_order'] + "0",
                        "0" + "01"*self.sequence_params['lmt_order'] + "0",
                        "0" + "10"*self.sequence_params['lmt_order'] + "1",
                        "0" + "01"*self.sequence_params['lmt_order'] + "1"]

        self.aisi_file.write('pathstosimulate ')
        for path in pathstosimulate:
            self.aisi_file.write(path + " ")
        self.aisi_file.write('\n')

        self.aisi_file.write('ignoredetuning {}\n'.format(self.simulation_params['ignoredetuning']))
        self.aisi_file.write('seed {}\n'.format(self.simulation_params['seed']))
        self.aisi_file.write('usedetvolselection {}\n'.format(self.simulation_params['usedetvolselection']))
        self.aisi_file.write('xdet {} {}\n'.format(self.simulation_params['xdet'][0], self.simulation_params['xdet'][1]))
        self.aisi_file.write('ydet {} {}\n'.format(self.simulation_params['ydet'][0], self.simulation_params['ydet'][1]))
        self.aisi_file.write('zdet {} {}\n'.format(self.simulation_params['zdet'][0], self.simulation_params['zdet'][1]))
        self.aisi_file.write('gslqagabserr {}\n'.format(self.simulation_params['gslqagabserr']))
        self.aisi_file.write('gslqagrelerr {}\n'.format(self.simulation_params['gslqagrelerr']))
        self.aisi_file.write('gslkinodeabserr {}\n'.format(self.simulation_params['gslkinodeabserr']))
        self.aisi_file.write('gslkinoderelerr {}\n'.format(self.simulation_params['gslkinoderelerr']))
        self.aisi_file.write('gslpulseodeabserr {}\n'.format(self.simulation_params['gslpulseodeabserr']))
        self.aisi_file.write('gslpulseoderelerr {}\n'.format(self.simulation_params['gslpulseoderelerr']))
        self.aisi_file.write('\n\n')

    def _write_sequence_params(self):
        self.aisi_file.write('# Sequence parameters\n')
        self.aisi_file.write('detectiontime {}\n\n'.format(self.sequence_params['detectiontime']))

    def _write_io_params(self):
        self.aisi_file.write('# IO parameters\n')
        self.aisi_file.write('printprobs {}\n'.format(self.io_params['printprobs']))
        self.aisi_file.write('printwavepackets {}\n\n'.format(self.io_params['printwavepackets']))

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
                
        elif self.sequence_params['sequencename'] == 'RB':
            if self.sequence_params['automaticdetuning'] == 1:
                if self.sequence_params['ultranarrow'] == 1:
                    self._write_auto_stepwise_detuning_ultranarrow_RB()
                #else:
                #    self._write_auto_stepwise_detuning()
            else:
                if (self.sequence_params['frequencychirp'] != 0) or (self.sequence_params['kchirp'] != 0):
                    if self.sequence_params['ultranarrow'] == 1:
                        self._write_chirped_sequence_ultranarrow_RB()
                    #else:
                    #    self._write_chirped_sequence()
                else:
                    raise ValueError("Only chirped or automatically detuned sequences are supported at the moment")
        
        else:
            raise ValueError("Sequence Name unknown")

    def _build_accel_pulses(self,
    t0_start,          # block start time
    vz0,               # initial vertical velocity (z)
    nkicks,            # number of π pulses in this block
    dt_pi,             # π-pulse duration
    dt_lmt,            # LMT spacing between π pulses
    recoil_sum,        # cumulative recoil (m/s) entering the block (for that arm)
    sign_seq,          # list of +1 / -1 of length nkicks, pulse direction per kick
    e_to_g_seq=None    # list of booleans (len nkicks); if None -> all False
    ):
        """
        Build an acceleration (LMT) block: start/end times and per-pulse (kz, omega),
        updating the recoil sum exactly like the original ultranarrow MZ code.

        Returns:
            dict with keys: 't0','t1','kz','omega','sign','recoil_sum'
        """
        if e_to_g_seq is None:
            e_to_g_seq = [False] * nkicks
        assert len(sign_seq) == nkicks
        assert len(e_to_g_seq) == nkicks

        start_times, end_times = [], []
        kz_vals, omega_vals, signs = [], [], []

        for i in range(nkicks):
            # Pulse timing
            t_i = t0_start + i * (dt_pi + dt_lmt)
            start_times.append(t_i)
            end_times.append(t_i + dt_pi)

            # Classical + recoil velocity at pulse i
            v_t = vz0 - g * t_i
            v_tot = v_t + recoil_sum

            # Direction & detuning branch as in your current code
            s = sign_seq[i]
            is_e_to_g = e_to_g_seq[i]

            if s == +1:
                omega_i = detuning(v_tot, is_e_to_g)
                kz_i = omega_i / c
                # emit +kz
                kz_vals.append(kz_i)
                omega_vals.append(omega_i)
                # update recoil by +ħ kz_i / m
                recoil_sum += hbar * kz_i / m
            else:  # s == -1
                omega_i = detuning(-v_tot, is_e_to_g)
                kz_i = omega_i / c
                # emit -kz_i (note: update recoil with +ħ*kz_i/m, same as your code)
                kz_vals.append(-kz_i)
                omega_vals.append(omega_i)
                recoil_sum += hbar * kz_i / m

            signs.append(s)

        return {
            't0': start_times,
            't1': end_times,
            'kz': kz_vals,
            'omega': omega_vals,
            'sign': signs,
            'recoil_sum': recoil_sum
        }

    def _build_decel_pulses(
    self,
    t0_start,        # block start time (already includes any dead time T)
    vz0,             # initial vertical (z) velocity
    nkicks,          # number of π pulses in this block
    dt_pi,           # π duration
    dt_lmt,          # LMT spacing between π pulses
    recoil_sum,      # cumulative recoil (m/s) entering this block (same arm)
    sign_seq,        # list of +1/-1 of length nkicks (laser direction per kick)
    e_to_g_seq=None, # list[bool] length nkicks; if None => all False
    use_global_k_for_up=True  # reproduce original block-2 behavior exactly
    ):
        """
        Build a *deceleration* LMT block (your block 2):
        - timings: t_i = t0_start + i*(dt_pi + dt_lmt)
        - detuning branch identical to your code
        - recoil update matches your original:
            if sign==+1: recoil_sum -= ħ * (kz if use_global_k_for_up else kz_i) / m
            if sign==-1: recoil_sum -= ħ * kz_i / m
        Returns:
            dict: 'kz','omega','sign','recoil_sum' (times are implied by t0_start/durations)
        """
        if e_to_g_seq is None:
            e_to_g_seq = [False] * nkicks
        assert len(sign_seq) == nkicks
        assert len(e_to_g_seq) == nkicks

        kz_vals, omega_vals, signs = [], [], []

        for i in range(nkicks):
            t_i = t0_start + i * (dt_pi + dt_lmt)

            # classical + current-recoil velocity for this arm
            v_t = vz0 - g * t_i
            vtot = v_t + recoil_sum

            s = sign_seq[i]
            is_e_to_g = e_to_g_seq[i]

            if s == +1:
                # same detuning branch as acceleration, but recoil UPDATE is negative (decelerate)
                omega_i = detuning(vtot, is_e_to_g)
                kz_i = omega_i / c
                kz_vals.append(kz_i)   # emit +kz_i
                omega_vals.append(omega_i)

                # IMPORTANT: match your original block-2 code exactly:
                #    sum_recoil_upper -= ħ * kz / m   (uses global detuning-less kz)
                # If you want fully consistent detuned recoil in the future, set use_global_k_for_up=False.
                if use_global_k_for_up:
                    recoil_sum -= hbar * kz / m
                else:
                    recoil_sum -= hbar * kz_i / m
            else:
                # s == -1: use opposite velocity in detuning; emit -kz_i; subtract detuned recoil
                omega_i = detuning(-vtot, is_e_to_g)
                kz_i = omega_i / c
                kz_vals.append(-kz_i)
                omega_vals.append(omega_i)
                recoil_sum -= hbar * kz_i / m

            signs.append(s)

        return {
            "kz": kz_vals,
            "omega": omega_vals,
            "sign": signs,
            "recoil_sum": recoil_sum,
        }


    def _write_auto_stepwise_detuning_ultranarrow_MZ(self):
        # --- unpack & preliminaries (unchanged) ---
        v0 = self.cloud_params['v0'][2]
        n = self.sequence_params['lmt_order']
        omega_chirp = 0
        kchirp = 0

        dt_bs = pi / (2 * self.pulse_params['rabi_freq'])
        dt_lmt = self.sequence_params['dt_lmt']
        pulse_duration = np.pi / (self.pulse_params['rabi_freq'])
        dt_acc = dt_lmt + pulse_duration
        if n == 1:
            dt_acc = mp.inf

        z0 = self.cloud_params['x0'][2]
        t_init = self.sequence_params['t_init']
        lmt_order = self.sequence_params['lmt_order']
        T = self.sequence_params['interrogation_time']
        rabi_freq = self.pulse_params['rabi_freq']
        wtype = self.pulse_params['wtype']
        phi0 = self.pulse_params['phi0']
        kx_psr = self.pulse_params['kx_psr']
        ky_psr = self.pulse_params['ky_psr']
        beam_radius = self.pulse_params['beam_radius']
        baseline = self.pulse_params['baseline']
        zernike_params = self.pulse_params['zernike_params']

        if lmt_order == 1:
            assert dt_lmt == 0
        else:
            assert dt_lmt > 0

        dt1 = dt_lmt
        dt2 = dt_lmt
        dt3 = dt_lmt
        dt4 = dt_lmt

        assert lmt_order % 2 == 1
        nlmt = int((lmt_order - 1))  # number of π pulses per LMT block

        # --- timing (unchanged structure) ---
        dt_bs = pi / (2 * rabi_freq)
        dt_pi = pi / (rabi_freq)
        T = T - 2 * (n - 1) * (dt_lmt + dt_pi)

        lmt_pulse_index = np.arange(0, nlmt + 1)
        t_start_shifted = lmt_pulse_index * (dt_pi + dt_lmt)
        t_end_shifted = t_start_shifted + dt_pi
        t_tot = t_end_shifted[-1] if nlmt > 1 else mp.mpf('0')

        t_bs1 = t_init
        t0 = dt_bs + t_init + dt1                   # start block 1 (accel upper arm)
        t1 = t0 + t_tot + T                         # start block 2 (decel upper arm) — dead time T included
        t_pi_t = t1 + t_tot + dt2                   # central mirror π
        t2 = t_pi_t + dt_pi + dt3                   # start block 3 (accel lower arm)
        t3 = t2 + t_tot + T                         # start block 4 (decel lower arm)
        t_bs2 = t3 + t_tot + dt4                    # final BS

        # --- build start/end times exactly as before ---
        start_times, end_times = [], []
        # initial π/2
        start_times.append(t_bs1); end_times.append(t_bs1 + dt_bs)
        # block 1
        for i in range(nlmt):
            start_times.append(t0 + t_start_shifted[i])
            end_times.append(t0 + t_end_shifted[i])
        # block 2
        for i in range(nlmt):
            start_times.append(t1 + t_start_shifted[i])
            end_times.append(t1 + t_end_shifted[i])
        # mirror
        start_times.append(t_pi_t); end_times.append(t_pi_t + dt_pi)
        # block 3
        for i in range(nlmt):
            start_times.append(t2 + t_start_shifted[i])
            end_times.append(t2 + t_end_shifted[i])
        # block 4
        for i in range(nlmt):
            start_times.append(t3 + t_start_shifted[i])
            end_times.append(t3 + t_end_shifted[i])
        # final π/2
        start_times.append(t_bs2); end_times.append(t_bs2 + dt_bs)

        # --- directions and transitions (unchanged) ---
        sign = []
        for i in range(n):          # block 1 (+ initial BS counted separately in detuning arrays)
            sign.append((-1) ** i)
        for i in range(2 * n - 1):  # block 2 + mirror index
            sign.append((-1) ** i)
        for i in range(n):          # block 3/4 + final BS
            sign.append((-1) ** i)

        is_e_to_g = [False] + [True, False] * (2 * n - 2) + [True] + [True, False] * (2 * n - 2) + [False]

        # --- detuning arrays ---
        kz_detuned_values = []
        omega0_detuned_values = []

        # initial π/2
        omega_ = detuning(v0, is_e_to_g[0])
        kz_ = omega_ / c
        kz_detuned_values.append(kz_)
        omega0_detuned_values.append(omega_)
        sum_recoil_upper = hbar * kz_ / m
        sum_recoil_lower = 0

        # ---- BLOCK 1: accelerate upper arm ----
        nk_b1 = nlmt
        sign_b1 = [sign[i] for i in range(1, n)]            # indices 1..n-1
        e2g_b1  = [is_e_to_g[i] for i in range(1, n)]
        b1 = self._build_accel_pulses(
            t0_start=t0, vz0=v0, nkicks=nk_b1,
            dt_pi=dt_pi, dt_lmt=dt_lmt,
            recoil_sum=sum_recoil_upper,
            sign_seq=sign_b1, e_to_g_seq=e2g_b1
        )
        kz_detuned_values.extend(b1["kz"])
        omega0_detuned_values.extend(b1["omega"])
        sum_recoil_upper = b1["recoil_sum"]

        # ---- BLOCK 2: decelerate upper arm ----
        nk_b2 = nlmt
        sign_b2 = [sign[i] for i in range(n, 2 * n - 1)]    # indices n..2n-2
        e2g_b2  = [is_e_to_g[i] for i in range(n, 2 * n - 1)]
        b2 = self._build_decel_pulses(
            t0_start=t1, vz0=v0, nkicks=nk_b2,
            dt_pi=dt_pi, dt_lmt=dt_lmt,
            recoil_sum=sum_recoil_upper,
            sign_seq=sign_b2, e_to_g_seq=e2g_b2,
            use_global_k_for_up=True  # preserve exact original behavior
        )
        kz_detuned_values.extend(b2["kz"])
        omega0_detuned_values.extend(b2["omega"])
        sum_recoil_upper = b2["recoil_sum"]

        # ---- MIRROR pulse (index 2n-1) ----
        i_mirror = 2 * n - 1
        t0i = start_times[i_mirror]
        v_t = v0 - g * t0i
        # original code used last block-2 detuned kz magnitude for recoil here:
        last_kz_mag_b2 = abs(b2["kz"][-1]) if len(b2["kz"]) > 0 else abs(kz_detuned_values[-1])
        v_recoil = hbar * last_kz_mag_b2 / m
        vtot = v_t + v_recoil
        omega_ = detuning(vtot, is_e_to_g[i_mirror])
        kz_ = omega_ / c
        kz_detuned_values.append(kz_)
        omega0_detuned_values.append(omega_)
        sum_recoil_lower = hbar * kz_ / m

        # ---- BLOCK 3: accelerate lower arm (note: not is_e_to_g) ----
        nk_b3 = nlmt
        sign_b3 = [sign[i] for i in range(2 * n, 3 * n - 1)]   # indices 2n..3n-2
        e2g_b3  = [not is_e_to_g[i] for i in range(2 * n, 3 * n - 1)]
        b3 = self._build_accel_pulses(
            t0_start=t2, vz0=v0, nkicks=nk_b3,
            dt_pi=dt_pi, dt_lmt=dt_lmt,
            recoil_sum=sum_recoil_lower,
            sign_seq=sign_b3, e_to_g_seq=e2g_b3
        )
        kz_detuned_values.extend(b3["kz"])
        omega0_detuned_values.extend(b3["omega"])
        sum_recoil_lower = b3["recoil_sum"]

        # ---- BLOCK 4: decelerate lower arm (note: not is_e_to_g) ----
        nk_b4 = nlmt
        sign_b4 = [sign[i] for i in range(3 * n - 1, 4 * n - 2)]   # indices 3n-1..4n-3
        e2g_b4  = [not is_e_to_g[i] for i in range(3 * n - 1, 4 * n - 2)]
        b4 = self._build_decel_pulses(
            t0_start=t3, vz0=v0, nkicks=nk_b4,
            dt_pi=dt_pi, dt_lmt=dt_lmt,
            recoil_sum=sum_recoil_lower,
            sign_seq=sign_b4, e_to_g_seq=e2g_b4,
            use_global_k_for_up=False  # original block-4 subtracts detuned kz_i for both signs
        )
        kz_detuned_values.extend(b4["kz"])
        omega0_detuned_values.extend(b4["omega"])
        sum_recoil_lower = b4["recoil_sum"]

        # ---- final beam splitter ----
        t0i = start_times[-1]
        v_t = v0 - g * t0i
        vtot = v_t
        omega_ = detuning(vtot, is_e_to_g[-1])
        kz_ = omega_ / c
        kz_detuned_values.append(kz_)
        omega0_detuned_values.append(omega_)

        # ---- transverse components and output (unchanged) ----
        kx = np.zeros(3 + 4 * nlmt)
        ky = np.zeros(3 + 4 * nlmt)
        kx[-1] = kx_psr
        ky[-1] = ky_psr

        self.aisi_file.write("# Pulse parameters\n")
        self.aisi_file.write("t0 " + " ".join(str(t) for t in start_times) + "\n")
        self.aisi_file.write("t1 " + " ".join(str(t) for t in end_times) + "\n")
        self.aisi_file.write("kx " + " ".join(str(x) for x in kx) + "\n")
        self.aisi_file.write("ky " + " ".join(str(y) for y in ky) + "\n")
        self.aisi_file.write("kz " + " ".join(str(k) for k in kz_detuned_values[:(3 + 4 * nlmt)]) + "\n")
        self.aisi_file.write("omega " + " ".join(str(w) for w in omega0_detuned_values[:(3 + 4 * nlmt)]) + "\n")
        self.aisi_file.write("rabifreq " + " ".join(str(rabi_freq / (2 * pi)) for _ in range(3 + 4 * nlmt)) + "\n")
        self.aisi_file.write("wtype " + " ".join(wtype for _ in range(3 + 4 * nlmt)) + "\n")
        self.aisi_file.write("phi0 " + " ".join(str(phi0) if i == 2 + 4 * nlmt else "0" for i in range(3 + 4 * nlmt)) + "\n")
        self.aisi_file.write("kxchirp " + " ".join("0" for _ in range(3 + 4 * nlmt)) + "\n")
        self.aisi_file.write("kychirp " + " ".join("0" for _ in range(3 + 4 * nlmt)) + "\n")
        self.aisi_file.write("kzchirp " + " ".join(str(-sign[i] * kchirp) for i in range(3 + 4 * nlmt)) + "\n")
        self.aisi_file.write("frequencychirp " + " ".join(str(-sign[i] * omega_chirp) for i in range(3 + 4 * nlmt)) + "\n")
        self.aisi_file.write("waist " + " ".join(str(self.pulse_params['waist']) for _ in range(3 + 4 * nlmt)) + "\n")
        # NOTE: original wrote focallength for 4+4*nlmt; keep that quirk intact:
        self.aisi_file.write("focallength " + " ".join(str(self.pulse_params['focallength']) for _ in range(4 + 4 * nlmt)) + "\n")
        self.aisi_file.write("zlaser " + " ".join(str(self.pulse_params['zupwardlaser']) if sign[i] == 1 else str(self.pulse_params['zdownwardlaser']) for i in range(3 + 4 * nlmt)) + "\n")
        self.aisi_file.write("beamradius " + " ".join(str(beam_radius) for _ in range(3 + 4 * nlmt)) + "\n")
        self.aisi_file.write("baseline " + " ".join(str(baseline) for _ in range(3 + 4 * nlmt)) + "\n")

        for zernike_noll_index in zernike_params.keys():
            self.aisi_file.write(f"zernikecoeff_{zernike_noll_index} ")
            for i in range(0, int(1 + (lmt_order - 1) / 2)):
                self.aisi_file.write(str(zernike_params[zernike_noll_index][0] if sign[i] == 1 else zernike_params[zernike_noll_index][3]) + " ")
            for i in range(lmt_order):
                s = sign[i + int(1 + (lmt_order - 1) / 2)]
                self.aisi_file.write(str(zernike_params[zernike_noll_index][1] if s == 1 else zernike_params[zernike_noll_index][4]) + " ")
            for i in range(int(1 + (lmt_order - 1) / 2)):
                s = sign[i + int(1 + (lmt_order - 1) / 2) + lmt_order]
                self.aisi_file.write(str(zernike_params[zernike_noll_index][2] if s == 1 else zernike_params[zernike_noll_index][5]) + " ")
            self.aisi_file.write("\n")

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
        zernike_params = self.pulse_params['zernike_params']

        # timing constants (match your single-loop conventions)
        dt_bs = pi / (2 * rabi_freq)
        dt_pi = pi / (    rabi_freq)

        # same per-block local spacings as your ultranarrow MZ
        dt1 = dt_lmt; dt2 = dt_lmt; dt3 = dt_lmt; dt4 = dt_lmt

        # LMT block internal span (sum of π + gaps between them)
        if nlmt > 1:
            t_tot = (np.arange(0, nlmt+1) * (dt_pi + dt_lmt))[-1] + dt_pi
        elif nlmt == 1:
            t_tot = dt_pi
        else:
            t_tot = mp.mpf('0')

        # build symmetric interrogation list
        ndiamonds=L
        L = len(T_base)
        assert L >= 1
        if ndiamonds==1:
            T_full = [T_base[0]]        # single diamond
        else:
            if ndiamonds%2==0:
                T_full = self._make_palindrome(T_base, include_center_twice=True)
            else:
                T_full = self._make_palindrome(T_base, include_center_twice=False)
            #T_full = list(T_base) + list(T_base[-2::-1])  # [T1,...,TL-1, TL, TL-1,...,T1]
        D = len(T_full)                  # number of diamonds in the full chain

        # ----- compute the effective interrogation times
        for i in range(len(T_full)):
            T_full[i] = T_full[i] - 2*(n-1)*(dt_lmt + dt_pi)

        # ----- pulse arrays
        start_times, end_times = [], []
        kz_vals, omega_vals    = [], []

        # ---- helpers (local) ----

        def _block_signs(start_sign, nkicks):
            # alternate +1, -1, +1, ... starting from start_sign
            return [start_sign * ((-1) ** k) for k in range(nkicks)]

        def _block_e2g(nkicks, flip=False):
            # matches your pattern: starts True and alternates per kick
            seq = [(k % 2 == 0) for k in range(nkicks)]
            return [ (not x) if flip else x for x in seq ]

        def _emit_block_times(t0_start, nkicks):
            t0s = [t0_start + k * (dt_pi + dt_lmt) for k in range(nkicks)]
            t1s = [t + dt_pi for t in t0s]
            start_times.extend(t0s); end_times.extend(t1s)

        def _accel_block(t0_start, vz0, recoil_sum, start_sign, e2g_flip=False):
            signs = _block_signs(start_sign, nlmt)
            e2g   = _block_e2g(nlmt, flip=e2g_flip)
            _emit_block_times(t0_start, nlmt)
            local_kz, local_om = [], []
            for k in range(nlmt):
                t_i   = t0_start + k * (dt_pi + dt_lmt)
                v_t   = vz0 - g * t_i
                v_tot = v_t + recoil_sum
                s     = signs[k]
                is_e  = e2g[k]
                if s == +1:
                    om = detuning(v_tot, is_e); kz_i = om / c
                    local_kz.append(kz_i);  local_om.append(om)
                    recoil_sum += hbar * kz_i / m
                else:
                    om = detuning(-v_tot, is_e); kz_i = om / c
                    local_kz.append(-kz_i); local_om.append(om)
                    recoil_sum += hbar * kz_i / m
            return recoil_sum, local_kz, local_om

        def _decel_block(t0_start, vz0, recoil_sum, start_sign, e2g_flip=False, use_global_k_for_up=False):
            # use_global_k_for_up=True reproduces your block-2: subtract ħ*kz/m for s=+1
            signs = _block_signs(start_sign, nlmt)
            e2g   = _block_e2g(nlmt, flip=e2g_flip)
            _emit_block_times(t0_start, nlmt)
            local_kz, local_om = [], []
            for k in range(nlmt):
                t_i   = t0_start + k * (dt_pi + dt_lmt)
                v_t   = vz0 - g * t_i
                v_tot = v_t + recoil_sum
                s     = signs[k]
                is_e  = e2g[k]
                if s == +1:
                    om = detuning(v_tot, is_e); kz_i = om / c
                    local_kz.append(kz_i);  local_om.append(om)
                    recoil_sum -= hbar * (kz if use_global_k_for_up else kz_i) / m
                else:
                    om = detuning(-v_tot, is_e); kz_i = om / c
                    local_kz.append(-kz_i); local_om.append(om)
                    recoil_sum -= hbar * kz_i / m
            return recoil_sum, local_kz, local_om

        # ---- time cursor and beam-splitters
        t = self.sequence_params['t_init']
        # initial π/2
        start_times.append(t); end_times.append(t + dt_bs)
        # detuning for initial π/2 (as before)
        om = detuning(v0, False); kz_i = om / c
        kz_vals.append(kz_i); omega_vals.append(om)
        sum_recoil_upper = hbar * kz_i / m
        sum_recoil_lower = mp.mpf('0')
        t += dt_bs

        # Per your single-loop pattern with odd n, the block start signs are:
        #   B1: +1   (accel upper),   B2: -1   (decel upper, uses global kz for s=+1),
        #   B3: +1   (accel lower, flip e2g),  B4: +1   (decel lower, detuned kz for both signs)
        start_sign_B1 = -1
        start_sign_B2 = +1              # because n is odd
        start_sign_B3 = -1
        start_sign_B4 = +1

        # walk diamonds
        for Ti in T_full:
            # --- Block 1 (upper accel)
            t += dt1
            sum_recoil_upper, kz_b1, om_b1 = _accel_block(t, v0, sum_recoil_upper, start_sign_B1, e2g_flip=False)
            kz_vals.extend(kz_b1); omega_vals.extend(om_b1)
            t += t_tot

            # wait Ti, then Block 2 (upper decel)
            t += Ti + dt2
            sum_recoil_upper, kz_b2, om_b2 = _decel_block(t, v0, sum_recoil_upper, start_sign_B2, e2g_flip=False, use_global_k_for_up=True)
            kz_vals.extend(kz_b2); omega_vals.extend(om_b2)
            t += t_tot

            # mirror π
            start_times.append(t); end_times.append(t + dt_pi)
            # mirror recoil: use |last kz| from block 2 (matches your original)
            last_kz_mag = abs(kz_b2[-1]) if len(kz_b2) else abs(kz_vals[-1])
            v_recoil = hbar * last_kz_mag / m
            v_t = v0 - g * t
            om = detuning(v_t + v_recoil, True); kz_i = om / c
            kz_vals.append(kz_i); omega_vals.append(om)
            sum_recoil_lower = hbar * kz_i / m
            t += dt_pi + dt3

            # --- Block 3 (lower accel, flip e2g)
            sum_recoil_lower, kz_b3, om_b3 = _accel_block(t, v0, sum_recoil_lower, start_sign_B3, e2g_flip=True)
            kz_vals.extend(kz_b3); omega_vals.extend(om_b3)
            t += t_tot

            # wait Ti, then Block 4 (lower decel, flip e2g)
            t += Ti + dt4
            sum_recoil_lower, kz_b4, om_b4 = _decel_block(t, v0, sum_recoil_lower, start_sign_B4, e2g_flip=True, use_global_k_for_up=False)
            kz_vals.extend(kz_b4); omega_vals.extend(om_b4)
            t += t_tot
            # next diamond continues immediately

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
        self.aisi_file.write("t0 " + " ".join(str(x) for x in start_times) + "\n")
        self.aisi_file.write("t1 " + " ".join(str(x) for x in end_times) + "\n")
        self.aisi_file.write("kx " + " ".join(str(x) for x in kx) + "\n")
        self.aisi_file.write("ky " + " ".join(str(y) for y in ky) + "\n")
        self.aisi_file.write("kz " + " ".join(str(k) for k in kz_vals) + "\n")
        self.aisi_file.write("omega " + " ".join(str(w) for w in omega_vals) + "\n")
        self.aisi_file.write("rabifreq " + " ".join(str(rabi_freq/(2*pi)) for _ in range(N)) + "\n")
        self.aisi_file.write("wtype " + " ".join(wtype for _ in range(N)) + "\n")
        # phase on the very last pulse (keep your convention)
        self.aisi_file.write("phi0 " + " ".join(str(phi0) if i == (N-1) else "0" for i in range(N)) + "\n")
        # zero chirps here
        self.aisi_file.write("kxchirp " + " ".join("0" for _ in range(N)) + "\n")
        self.aisi_file.write("kychirp " + " ".join("0" for _ in range(N)) + "\n")
        self.aisi_file.write("kzchirp " + " ".join("0" for _ in range(N)) + "\n")
        self.aisi_file.write("frequencychirp " + " ".join("0" for _ in range(N)) + "\n")
        self.aisi_file.write("waist " + " ".join(str(self.pulse_params['waist']) for _ in range(N)) + "\n")
        self.aisi_file.write("focallength " + " ".join(str(self.pulse_params['focallength']) for _ in range(N)) + "\n")
        # z-laser per sign (we can infer from kz sign we emitted):
        self.aisi_file.write("zlaser " + " ".join(
            str(self.pulse_params['zupwardlaser']) if (mp.sign(kz_vals[i]) >= 0) else str(self.pulse_params['zdownwardlaser'])
            for i in range(N)
        ) + "\n")
        self.aisi_file.write("beamradius " + " ".join(str(beam_radius) for _ in range(N)) + "\n")
        self.aisi_file.write("baseline " + " ".join(str(baseline) for _ in range(N)) + "\n")

        # Zernike (reuse your 3-block grouping per diamond; here we emit a simple per-pulse up/down split)
        for zidx in zernike_params.keys():
            c1_up, c2_up, c3_up, c1_dn, c2_dn, c3_dn = zernike_params[zidx]
            # Map by segment within each diamond; to keep this concise, emit c2_up/dn for all LMT π and c1 for BS/mirror:
            # (You can elaborate to mirror exactly your block-by-block coefficient layout if needed.)
            out = []
            # classify each pulse: BS/mirror vs LMT by duration = dt_bs or dt_pi
            for i in range(N):
                dur = end_times[i] - start_times[i]
                is_bs = abs(dur - dt_bs) < 1e-30
                is_pi = abs(dur - dt_pi) < 1e-30
                if is_bs:
                    out.append(str(c1_up if kz_vals[i] >= 0 else c1_dn))
                elif is_pi:
                    out.append(str(c2_up if kz_vals[i] >= 0 else c2_dn))
                else:
                    out.append("0")
            self.aisi_file.write(f"zernikecoeff_{zidx} " + " ".join(out) + "\n")



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

        print(sign)

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


        
