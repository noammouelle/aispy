import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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