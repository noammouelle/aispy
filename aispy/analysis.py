import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from mpmath import mp, mpf
from math import pi

def load_data(filename):
    with h5py.File(filename,"r") as file:
        states = file["states"][:]
        positions = file["positions"][:]
        velocities = file["velocities"][:]
        phase_shifts = file["phaseShifts"][:]
        interference_flag = file["interferingFlag"][:]

    df = pd.DataFrame({"states":states, "x":positions[:,0], "y":positions[:,1], "z":positions[:,2],
                       "vx":velocities[:,0], "vy":velocities[:,1], "vz":velocities[:,2],
                       "phase_shifts":phase_shifts, "interference_flag":interference_flag})
    return df

def get_params_dict(file_path):
    """
    Returns the params dict from the input.aisi file
    """
    param_dict = {
        'cloud_params': {},
        'potential_params': {},
        'sequence_params': {},
        'pulse_params': {},
        'simulation_params': {}
    }

    # Open and read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse each line based on keywords
    for line in lines:
        line = line.strip()
        
        # Ignore comments or empty lines
        if line.startswith("#") or not line:
            continue

        # Cloud parameters
        if line.startswith("natoms"):
            param_dict['cloud_params']['natoms'] = int(line.split()[1])
        elif line.startswith("sigma"):
            param_dict['cloud_params']['sigma'] = float(line.split()[1])
        elif line.startswith("temp"):
            param_dict['cloud_params']['temp'] = float(line.split()[1])
        elif line.startswith("x0"):
            param_dict['cloud_params']['x0'] = [float(x) for x in line.split()[1:]]
        elif line.startswith("v0"):
            param_dict['cloud_params']['v0'] = [float(x) for x in line.split()[1:]]

        # Potential parameters
        elif line.startswith("utype"):
            param_dict['potential_params']['utype'] = line.split()[1]

        # Simulation parameters
        elif line.startswith("amplitudethreshold"):
            param_dict['simulation_params']['amplitudethreshold'] = float(line.split()[1])

        # Sequence parameters
        elif line.startswith("detectiontime"):
            param_dict['sequence_params']['detectiontime'] = mpf(line.split()[1])

        # Pulse parameters
        elif line.startswith("rabifreq"):
            param_dict['pulse_params']['rabi_freq'] = 2 * pi * mpf(line.split()[1])
        elif line.startswith("wtype"):
            param_dict['pulse_params']['wtype'] = line.split()[1]
        elif line.startswith("phi0"):
            param_dict['pulse_params']['phi0'] = float(line.split()[1])
        elif line.startswith("kx"):
            param_dict['pulse_params']['kx_psr'] = int(line.split()[3])

    return param_dict