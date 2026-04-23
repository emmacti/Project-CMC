#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

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
    """
    Q3.2 – Sweep stretch feedback strength w_ipsi and measure performance.

    For each w_ipsi in [-3, 17], this script:
    - runs a simulation (headless/fast)
    - computes forward speed + CoT
    - computes mean neural frequency + amplitude (from ML-MR motor outputs)
    - saves summary plots vs w_ipsi in `results/`
    """
    pylog.set_level('warning')
    os.makedirs(PLOT_PATH, exist_ok=True)

    w_ipsi_range = np.linspace(-3, 17, 11)

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

    # Collect metrics per w_ipsi
    speed = np.zeros(len(w_ipsi_range), dtype=float)
    cot = np.zeros(len(w_ipsi_range), dtype=float)
    neur_freq_mean = np.zeros(len(w_ipsi_range), dtype=float)
    neur_amp_mean = np.zeros(len(w_ipsi_range), dtype=float)

    def _token(x: float) -> str:
        # Must match simulate._as_filename_token for floats
        return f"{float(x):0.3f}"

    for i, w in enumerate(w_ipsi_range):
        hdf5_path = os.path.join(BASE_PATH, f"simulation_w_ipsi{_token(w)}.hdf5")
        ctrl_path = os.path.join(BASE_PATH, f"controller_w_ipsi{_token(w)}.pkl")

        with h5py.File(hdf5_path, "r") as f:
            times = f['times'][:]
            links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
            joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

        links_pos = links[:, :, 7:10]
        links_vel = links[:, :, 14:17]
        joints_vel = joints[:, :, 1]
        joints_tau = joints[:, :, 2]

        v_fwd, _v_lat = compute_mechanical_speed(links_positions=links_pos, links_velocities=links_vel)
        _energy, c = compute_mechanical_energy_and_cot(
            times=times,
            links_positions=links_pos,
            joints_torques=joints_tau,
            joints_velocities=joints_vel,
        )
        speed[i] = float(v_fwd)
        cot[i] = float(c)

        # Neural metrics: compute from controller motor outputs (ML-MR)
        with open(ctrl_path, "rb") as f:
            controller_data = pickle.load(f)
        state = controller_data["state"]
        n_total = state.shape[1]
        if n_total % 3 != 0:
            raise ValueError("Unexpected controller state width in exercise3_2 sweep.")
        n_osc = n_total // 3
        motor_storage = state[:, 2*n_osc:3*n_osc]
        ml = motor_storage[:, 0::2]
        mr = motor_storage[:, 1::2]
        neural = ml - mr

        neural_s = filter_signals(times=times, signals=neural)
        freqs, _freq_bins, amps = compute_frequency_amplitude_fft(times=times, smooth_signals=neural_s)
        neur_freq_mean[i] = float(np.mean(freqs))
        neur_amp_mean[i] = float(np.mean(amps))

    # Plots: metrics vs w_ipsi (Q3.2)
    def _plot(y, ylabel, fname):
        plt.figure(figsize=(6.5, 4))
        plt.plot(w_ipsi_range, y, 'o-', lw=1.8)
        plt.xlabel(r'$w^{ipsi}$')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, fname), dpi=200)
        plt.close()

    _plot(speed, 'Forward speed (m/s)', 'exercise3_2_speed_vs_wipsi.png')
    _plot(cot, 'CoT (J/m)', 'exercise3_2_cot_vs_wipsi.png')
    _plot(neur_freq_mean, 'Mean neural frequency (Hz)', 'exercise3_2_neur_freq_vs_wipsi.png')
    _plot(neur_amp_mean, 'Mean neural amplitude', 'exercise3_2_neur_amp_vs_wipsi.png')

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise3_2(plot=True)
