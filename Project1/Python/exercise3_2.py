#!/usr/bin/env python3


import os
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    filter_signals,
)
from simulate import run_multiple

# Multiprocessing
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
MAX_WORKERS = 8  # adjust based on your hardware capabilities


BASE_PATH = 'logs/exercise3_2/'
PLOT_PATH = 'results'

# CPG parameters
DRIVE_LEFT = 3
DRIVE_RIGHT = 3
DRIVE_LOW = 1
DRIVE_HIGH = 5
A_RATE = np.ones(8) * 3
OFFSET_FREQ = np.ones(8) * 1
OFFSET_AMP = np.ones(8) * 0.5
G_FREQ = np.ones(8) * 0.5
G_AMP = np.ones(8) * 0.25
PHASELAG = np.ones(7) * np.pi * 2 / 8
COUPLING_WEIGHTS_ROSTRAL = 5
COUPLING_WEIGHTS_CAUDAL = 5
COUPLING_WEIGHTS_CONTRA = 10
# random init phases for 16 oscillators for 8 joints
INIT_PHASE = np.random.default_rng(
    seed=42).uniform(0.0, 2 * np.pi, size=16)

pylog.set_level('warning')
# pylog.set_level('critical') # suppress logging output in multi-processing

def exercise3_2(**kwargs):
    """ex3.2 main"""
    pylog.warning("TODO: 3.2 Explore the effect of stretch feedback on the metrics.")

    w_ipsi_range = np.linspace(0, 0, 5)

    controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': DRIVE_LEFT,
            'drive_right': DRIVE_RIGHT,
            'd_low': DRIVE_LOW,
            'd_high': DRIVE_HIGH,
            'a_rate': A_RATE,
            'offset_freq': OFFSET_FREQ,
            'offset_amp': OFFSET_AMP,
            'G_freq': G_FREQ,
            'G_amp': G_AMP,
            'PL': PHASELAG,
            'coupling_weights_rostral': COUPLING_WEIGHTS_ROSTRAL,
            'coupling_weights_caudal': COUPLING_WEIGHTS_CAUDAL,
            'coupling_weights_contra': COUPLING_WEIGHTS_CONTRA,
            'init_phase': INIT_PHASE,
        },
    }
    run_multiple(
        max_workers=MAX_WORKERS,
        controller=controller,
        base_path=BASE_PATH,
        parameter_grid={'w_ipsi': w_ipsi_range},
        common_kwargs={
            'fast': True,
            'headless': True,
            'runtime_n_iterations': 5001,
            'runtime_buffer_size': 5001,
        },
    )

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise3_2(plot=True)

