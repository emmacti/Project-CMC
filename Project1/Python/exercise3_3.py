#!/usr/bin/env python3

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
)
from simulate import runsim, run_multiple

pylog.set_level('warning')
# pylog.set_level('critical') # suppress logging output in multi-processing

BASE_PATH = 'logs/exercise3_3/'
PLOT_PATH = 'results'

# CPG parameters
DRIVE_LEFT = 3
DRIVE_RIGHT = 3
DRIVE_LOW = 1
DRIVE_HIGH = 5
A_RATE = np.ones(8) * 3
OFFSET_FREQ = np.ones(8) * 1
OFFSET_AMP = np.ones(8) * 0
G_FREQ = np.ones(8) * 0.5
G_AMP = np.ones(8) * 0.25
PHASELAG = np.ones(7) * np.pi * 2 / 8
COUPLING_WEIGHTS_ROSTRAL = 5
COUPLING_WEIGHTS_CAUDAL = 5
COUPLING_WEIGHTS_CONTRA = 10
# random init phases for 16 oscillators for 8 joints
INIT_PHASE = np.random.default_rng(
    seed=42).uniform(0.0, 2 * np.pi, size=16)
W_IPSI = 10.0

# disruption propabilities
DISRUPTION_P_SENSORS = 0.2
DISRUPTION_P_COUPLINGS = 0.2
RANDOM_SEED = 42
MAX_WORKERS = 8

def load_sim_data(hdf5_path, skip_start=500):
    """Load simulation sensor data and slice out initial transient."""
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f["times"][:]
        sensor_links = f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"][:]
        sensor_joints = f["FARMSLISTanimats"]["0"]["sensors"]["joints"]["array"][:]

    sim_times = sim_times[skip_start:]
    sensor_links_pos = sensor_links[skip_start:, :, 7:10]
    sensor_joints_pos = sensor_joints[skip_start:, :, 0]

    return sim_times, sensor_links_pos, sensor_joints_pos


def exercise3_3(**kwargs):
    """ex3.3 main"""
    pylog.warning("TODO: 3.3 Implement neural disruptions and compare with no disruption.")

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise3_3(plot=True)

