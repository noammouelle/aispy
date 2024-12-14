from mpmath import mp, sqrt, cos, cosh, sin, sinh
import numpy as np
import datetime

mp.dps = 34  # Set decimal precision to 34 digits

# define constants using mpmath for quad precision
c = mp.mpf('299792458')
g = mp.mpf('9.81')
R = mp.mpf('6.37e6')
m = mp.mpf('1.44e-25')
h = mp.mpf('6.62607015e-34')
kB = 1.381e-23
pi = mp.pi
hbar = h / (2 * pi)
rabi_freq = 2 * pi * mp.mpf('1e6')

omega0 = 2 * pi * mp.mpf("429228004229873.0")
kz     = omega0 / c

def v_uniform(t,v0):
    return v0 - g*t

def v(t,v0,z0):
    return -1/2 * (R - 2*z0) * sqrt(2*g/R) * sinh(sqrt(2*g/R) * t) + v0 * cosh(sqrt(2*g/R) * t)

def detuning(v):
    omega = omega0 / (1 - v/c)
    k = omega / c
    recoil = hbar * k **2 /(2*m)

    return omega + recoil

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
        self.aisi_file.write('temp {}\n'.format(self.cloud_params['temp']))
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
        self.aisi_file.write('ignoredetuning {}\n'.format(self.simulation_params['ignoredetuning']))
        self.aisi_file.write('seed {}\n'.format(self.simulation_params['seed']))
        self.aisi_file.write('usedetvolselection {}\n'.format(self.simulation_params['usedetvolselection']))
        self.aisi_file.write('xdet {} {}\n'.format(self.simulation_params['xdet'][0], self.simulation_params['xdet'][1]))
        self.aisi_file.write('ydet {} {}\n'.format(self.simulation_params['ydet'][0], self.simulation_params['ydet'][1]))
        self.aisi_file.write('zdet {} {}'.format(self.simulation_params['zdet'][0], self.simulation_params['zdet'][1]))
        self.aisi_file.write('\n\n')

    def _write_sequence_params(self):
        self.aisi_file.write('# Sequence parameters\n')
        self.aisi_file.write('detectiontime {}\n\n'.format(self.sequence_params['detectiontime']))

    def _write_io_params(self):
        self.aisi_file.write('# IO parameters\n')
        self.aisi_file.write('printprobs {}\n'.format(self.io_params['printprobs']))
        self.aisi_file.write('printwavepackets {}\n\n'.format(self.io_params['printwavepackets']))
        
    def _write_pulse_params(self):
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


        
