import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from mpmath import mp, mpf
from math import pi

def load_data(filename):
    if "_PROB.h5" in filename:
        with h5py.File(filename,"r") as file:
            states = file["states"][:]
            positions = file["positions"][:]
            velocities = file["velocities"][:]
            probabilities = file["probabilities"][:]
            interference_flag = file["interferingFlag"][:]

        df = pd.DataFrame({"states":states, "x":positions[:,0], "y":positions[:,1], "z":positions[:,2],
                        "vx":velocities[:,0], "vy":velocities[:,1], "vz":velocities[:,2],
                        "probabilities":probabilities, "interference_flag":interference_flag})
    else:
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

def concat_datasets(filepaths, output_filepath):
    # Empty list to collect DataFrames
    data_frames = []
    
    # Load each file, append DataFrame to the list
    for filepath in filepaths:
        df = load_data(filepath)
        data_frames.append(df)
    
    # Concatenate all DataFrames
    concatenated_df = pd.concat(data_frames, ignore_index=True)
    
    # Save concatenated data to a new HDF5 file in the same format
    with h5py.File(output_filepath, "w") as h5f:
        # Check if the data includes "probabilities" or "phase_shifts" column to set dataset structure
        if "probabilities" in concatenated_df.columns:
            h5f.create_dataset("states", data=concatenated_df["states"].values)
            h5f.create_dataset("positions", data=concatenated_df[["x", "y", "z"]].values)
            h5f.create_dataset("velocities", data=concatenated_df[["vx", "vy", "vz"]].values)
            h5f.create_dataset("probabilities", data=concatenated_df["probabilities"].values)
            h5f.create_dataset("interferingFlag", data=concatenated_df["interference_flag"].values)
        else:
            h5f.create_dataset("states", data=concatenated_df["states"].values)
            h5f.create_dataset("positions", data=concatenated_df[["x", "y", "z"]].values)
            h5f.create_dataset("velocities", data=concatenated_df[["vx", "vy", "vz"]].values)
            h5f.create_dataset("phaseShifts", data=concatenated_df["phase_shifts"].values)
            h5f.create_dataset("interferingFlag", data=concatenated_df["interference_flag"].values)

def get_params_dict(file_path):
    """
    Returns the params dict from the input.aisi file
    """
    param_dict = {
        'cloud_params': {},
        'potential_params': {},
        'sequence_params': {},
        'pulse_params': {
            'rabi_freq': [],
            'wtype': [],
            'phi0': [],
            'kx':[],
            'ky':[],
            'kz':[],
            'omega':[],
            't0':[],
            't1':[]
        },
        'simulation_params': {},
        'io_params': {}
    }

    # Open and read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

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
        elif line.startswith("coherencelength"):
            param_dict['simulation_params']['coherencelength'] = float(line.split()[1])
        elif line.startswith("usemcbranching"):
            param_dict['simulation_params']['usemcbranching'] = int(line.split()[1])
        
        # IO params
        elif line.startswith("printprobs"):
            param_dict['io_params']['printprobs'] = float(line.split()[1])

        # Sequence parameters
        elif line.startswith("detectiontime"):
            param_dict['sequence_params']['detectiontime'] = mpf(line.split()[1])

        # Pulse parameters
        elif line.startswith("rabifreq"):
            # Convert each frequency to 2 * pi * mpf and add to the list
            param_dict['pulse_params']['rabi_freq'].extend(
                [2 * pi * mpf(freq) for freq in line.split()[1:]]
            )
        
        elif line.startswith("wtype"):
            # Append each wtype as a string to the list
            param_dict['pulse_params']['wtype'].extend(line.split()[1:])
        
        elif line.startswith("phi0"):
            # Append each phi0 value as a float to the list
            param_dict['pulse_params']['phi0'].extend([float(phi) for phi in line.split()[1:]])
        
        elif line.startswith("kx"):
            # Append each kx value
            param_dict['pulse_params']['kx'].extend([float(k) for k in line.split()[1:]])

        elif line.startswith("ky"):
            # Append each kx value
            param_dict['pulse_params']['ky'].extend([float(k) for k in line.split()[1:]])

        elif line.startswith("kz"):
            # Append each kx value
            param_dict['pulse_params']['kz'].extend([float(k) for k in line.split()[1:]])

        elif line.startswith("t0"):
            param_dict['pulse_params']['t0'].extend([mpf(t) for t in line.split()[1:]])
        
        elif line.startswith("t1"):
            param_dict['pulse_params']['t1'].extend([mpf(t) for t in line.split()[1:]])

        elif line.startswith("omega"):
            param_dict['pulse_params']['omega'].extend([mpf(o) for o in line.split()[1:]])

    return param_dict