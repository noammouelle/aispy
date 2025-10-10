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
                    self._write_auto_stepwise_detuning_ultranarrow_MZ()
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

    def _write_auto_stepwise_detuning_ultranarrow_MZ(self):
        # get the initial vertical velocity
        v0 = self.cloud_params['v0'][2]

        # get the lmt order
        n = self.sequence_params['lmt_order']

        # set omega_chirp and kchirp to 0
        omega_chirp = 0
        kchirp = 0

        # get the duration of the lmt pulses
        dt_bs = pi / (2 * self.pulse_params['rabi_freq'])
        dt_lmt = self.sequence_params['dt_lmt']
        pulse_duration = np.pi / (self.pulse_params['rabi_freq'])

        dt_acc = dt_lmt + pulse_duration

        if n == 1:
            dt_acc = np.inf # sets the chirp to 0 for the first block

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
        
        dt1 = dt_lmt #+ np.pi/(2*rabi_freq) # makes the spacing same as for the pi pulses
        dt2 = dt_lmt
        dt3 = dt_lmt
        dt4 = dt_lmt #+ np.pi/(2*rabi_freq)

        assert(lmt_order%2 == 1)
        nlmt = int((lmt_order-1))

        # calculate the time sequence
        dt_bs = pi/(2*rabi_freq)
        dt_pi = pi/(rabi_freq)

        # compute the effective interrogation time
        T = T - 2*(n-1)*(dt_lmt + dt_pi)

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
        end_times.append(t_pi+ dt_pi)
        
        # LMT block 3
        for i in range(nlmt):
            start_times.append(t2 + t_start_shifted[i])
            end_times.append(t2 + t_end_shifted[i])

        # LMT block 4
        for i in range(nlmt):
            start_times.append(t3 + t_start_shifted[i])
            end_times.append(t3 + t_end_shifted[i])

        # final bs pulse
        start_times.append(t_bs2)
        end_times.append(t_bs2 + dt_bs)

        # compute the direction of the pulses
        sign = []
        for i in range(n):
            sign.append((-1)**i)
        for i in range(2*n-1): # block 2 and 3
            sign.append((-1)**i)
        for i in range(n): # block 4 and beam splitter pulse
            sign.append((-1)**i)

        # figure out which transitions are e->g
        is_e_to_g = [False] + [True, False]*(2*n-2)+ [True] + [True, False]*(2*n-2) + [False]

        # compute the kz and omega values for each block
        kz_detuned_values = []
        omega0_detuned_values = []

        # sum of the upper arm and lower arm recoil
        sum_recoil_upper = 0
        sum_recoil_lower = 0

        # pi/2 pulse
        omega_ = detuning(v0, is_e_to_g[0])
        kz_ = omega_ / c
        kz_detuned_values.append(kz_)
        omega0_detuned_values.append(omega_)

        sum_recoil_upper += hbar * kz_ / m

        # debug print
        #print("v0: ", v0)

        # first block is index 1 to n-1
        for i in range(1,n):
            # get the initial time of the pulse
            t0 = start_times[i]
            # compute the classical velocity at the time of the pulse
            v_t = v0 - g * t0
            # compute the velocity due to recoil
            v_recoil = sum_recoil_upper
            
            vtot = v_t + v_recoil

            # debug print
            #print("vtot: ", vtot)

            if sign[i] == 1:
                omega_ = detuning(vtot, is_e_to_g[i])
                kz_ = omega_ / c
                kz_detuned_values.append(kz_)
                omega0_detuned_values.append(omega_)

                sum_recoil_upper += hbar * kz_ / m
            else:
                omega_ = detuning(-vtot, is_e_to_g[i])
                kz_ = omega_ / c
                kz_detuned_values.append(-kz_)
                omega0_detuned_values.append(omega_)

                sum_recoil_upper += hbar * kz_ / m

        # second block is index n to 2n-2 (not including the mirror pulse)
        for i in range(n, 2*n-1):
            # get the initial time of the pulse
            t0 = start_times[i]
            # compute the classical velocity at the time of the pulse
            v_t = v0 - g * t0
            # compute the velocity due to recoil
            v_recoil = sum_recoil_upper
            vtot = v_t + v_recoil
            # debug print
            #print("vtot: ", vtot)
            if sign[i] == 1:
                omega_ = detuning(vtot, is_e_to_g[i])
                kz_ = omega_ / c
                kz_detuned_values.append(kz_)
                omega0_detuned_values.append(omega_)

                sum_recoil_upper -= hbar * kz / m # block 2 slows down the upper arm
            else:
                omega_ = detuning(-vtot, is_e_to_g[i])
                kz_ = omega_ / c
                kz_detuned_values.append(-kz_)
                omega0_detuned_values.append(omega_)

                sum_recoil_upper -= hbar * kz_ / m

        # mirror pulse
        i = 2*n-1
        # get the initial time of the pulse
        t0 = start_times[i]
        # compute the classical velocity at the time of the pulse
        v_t = v0 - g * t0
        # compute the velocity due to recoil, say hbar * kz / 2 m as adresses both arms
        v_recoil = hbar*kz_/ (1*m)
        vtot = v_t + v_recoil
        omega_ = detuning(vtot, is_e_to_g[i])
        kz_ = omega_ / c
        kz_detuned_values.append(kz_)
        omega0_detuned_values.append(omega_) 

        # debug print
        #print("vtot: ", vtot)

        # increment the recoil sum for the lower arm
        sum_recoil_lower = hbar * kz_ / m       

        # third block is index 2n to 3n-2
        for i in range(2*n, 3*n-1):
            # get the initial time of the pulse
            t0 = start_times[i]
            # compute the classical velocity at the time of the pulse
            v_t = v0 - g * t0
            # compute the velocity due to recoil
            v_recoil = sum_recoil_lower
            vtot = v_t + v_recoil
            # debug print
            #print("vtot: ", vtot)
            if sign[i] == 1:
                omega_ = detuning(vtot, not is_e_to_g[i]) # note the not here, dont know why it works like this
                kz_ = omega_ / c
                kz_detuned_values.append(kz_)
                omega0_detuned_values.append(omega_)

                sum_recoil_lower += hbar * kz_ / m
            else:
                omega_ = detuning(-vtot, not is_e_to_g[i])
                kz_ = omega_ / c
                kz_detuned_values.append(-kz_)
                omega0_detuned_values.append(omega_)

                sum_recoil_lower += hbar * kz_ / m

        # fourth block is index 3n-1 to 4n-3
        for i in range(3*n-1, 4*n-2):
            # get the initial time of the pulse
            t0 = start_times[i]
            # compute the classical velocity at the time of the pulse
            v_t = v0 - g * t0
            # compute the velocity due to recoil
            v_recoil = sum_recoil_lower
            vtot = v_t + v_recoil
            # debug print
            #print("vtot: ", vtot)
            if sign[i] == 1:
                omega_ = detuning(vtot, not is_e_to_g[i])
                kz_ = omega_ / c
                kz_detuned_values.append(kz_)
                omega0_detuned_values.append(omega_)

                sum_recoil_lower -= hbar * kz_ / m
            else:
                omega_ = detuning(-vtot, not is_e_to_g[i])
                kz_ = omega_ / c
                kz_detuned_values.append(-kz_)
                omega0_detuned_values.append(omega_)

                sum_recoil_lower -= hbar * kz_ / m

        # final beam splitter pulse
        # get the initial time of the pulse
        t0 = start_times[-1]
        # compute the classical velocity at the time of the pulse
        v_t = v0 - g * t0
        # compute the velocity due to recoil, say hbar * kz / 2 m as adresses both arms
        v_recoil = hbar*kz / (1*m)*0
        vtot = v_t + v_recoil
        omega_ = detuning(vtot, is_e_to_g[-1])
        kz_ = omega_ / c
        kz_detuned_values.append(kz_)
        omega0_detuned_values.append(omega_)
        # debug print
        #print("vtot: ", vtot)

        # transverse wavevector components
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

        # write the kz values for each block
        self.aisi_file.write("kz ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(kz_detuned_values[i]) + " ")
        self.aisi_file.write("\n")
        # write the detuned frequencies
        self.aisi_file.write("omega ")
        for i in range(3+4*nlmt):
            self.aisi_file.write(str(omega0_detuned_values[i]) + " ")
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
            if i == 2+4*nlmt:
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
            self.aisi_file.write(str(-sign[i]*kchirp) + " ")
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
        for i in range(4+4*nlmt):
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
            self.aisi_file.write("\n")

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


        
