#!/usr/bin/env python3
"""Run parameter sweeps for exercise 1.2 and plot metric heatmaps."""

import os
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from farms_core import pylog

from simulate import run_multiple
from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    compute_neural_phase_lags,
    filter_signals,
)

# Multiprocessing

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MAX_WORKERS = 8  # adjust based on your hardware capabilities

pylog.set_level('critical')

BASE_PATH = 'logs/exercise1_2/'
PLOT_PATH = 'results'
RECORDING = None  # disable recording for parallel runs


def get_metrics(twl, amp):
    """Compute mechanical metrics for a single parameter set."""
    # Load HDF5
    sim_result = BASE_PATH + \
        f'simulation_twl{twl:0.3f}_amp{amp:0.3f}.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    sensor_data_links_positions = sensor_data_links[:, :, 7:10]

    sensor_data_links_velocities = sensor_data_links[:, :, 14:17]
    sensor_data_joints_velocities = sensor_data_joints[:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[:, :, 2]

    speed_forward, _ = compute_mechanical_speed(
        links_positions=sensor_data_links_positions,
        links_velocities=sensor_data_links_velocities,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=sensor_data_links_positions,
        joints_torques=sensor_data_joints_torques,
        joints_velocities=sensor_data_joints_velocities,
    )

    # Load Controller
    controller_file = os.path.join(
        BASE_PATH,
        f"controller_twl{twl:0.3f}_amp{amp:0.3f}.pkl",
    )
    with open(controller_file, "rb") as f:
        controller_data = pickle.load(f)

    indices = controller_data["indices"]
    neural_signals = (
        controller_data["state"][:, indices['left_body_idx']]
        - controller_data["state"][:, indices['right_body_idx']]
    )
    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)
    signal_freqs, _, _ = compute_frequency_amplitude_fft(
        times=sim_times,
        smooth_signals=neural_signals_smoothed,
    )
    inds_couples = [[i, i + 1]
                    for i in range(neural_signals_smoothed.shape[1] - 1)]
    _, ipls_mean = compute_neural_phase_lags(
        times=sim_times,
        smooth_signals=neural_signals_smoothed,
        freqs=signal_freqs,
        inds_couples=inds_couples,
    )

    return speed_forward, cot, float(ipls_mean)


def exercise1_2(**kwargs):
    """
    Q1.2 (extended) – Sweep WaveController parameters and visualize performance.

    Runs a grid over:
    - amplitude A ∈ [1, 4]
    - total wave lag TWL ∈ [0.2, 1.5]
    with frequency fixed at f=1.5 Hz.

    Saves heatmaps to `results/`.
    """
    os.makedirs(PLOT_PATH, exist_ok=True)
    base_controller = {
        'loader': 'cmc_controllers.wave_controller.WaveController',
        'config': {
            'freq': 1.5,
            'twl': 1.0,
            'amp': 1.0}}
    # Parameter investigation grid (Question 1.3)
    # A ∈ [1.0, 4.0], TWL ∈ [0.2, 1.5], f = 1.5 Hz (fixed above)
    twl_range = np.linspace(0.2, 1.5, 7)
    amp_range = np.linspace(1.0, 4.0, 7)

    parameter_grid = {'twl': twl_range, 'amp': amp_range}

    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid,
        common_kwargs={'fast': True, 'headless': True},
    )

    # Analyze results: build metric heatmaps (forward speed, CoT, mean IPL)
    speed_map = np.zeros((len(twl_range), len(amp_range)))
    cot_map = np.zeros((len(twl_range), len(amp_range)))
    ipl_map = np.zeros((len(twl_range), len(amp_range)))

    for i, twl in enumerate(twl_range):
        for j, amp in enumerate(amp_range):
            v_fwd, cot, ipl = get_metrics(float(twl), float(amp))
            speed_map[i, j] = v_fwd
            cot_map[i, j] = cot
            ipl_map[i, j] = ipl

    def _save_heatmap(mat, title, fname, cmap='viridis', vmin=None, vmax=None, log=False):
        plt.figure(figsize=(6, 4.5))
        show = mat.copy()
        norm = None
        if log:
            show = np.clip(show, 1e-9, None)
            norm = colors.LogNorm(vmin=np.min(show), vmax=np.max(show))
        im = plt.imshow(
            show,
            origin='lower',
            aspect='auto',
            extent=[amp_range[0], amp_range[-1], twl_range[0], twl_range[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
        plt.colorbar(im, label=title)
        plt.xlabel('Amplitude A')
        plt.ylabel('Total wave lag (TWL)')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, fname), dpi=200)

    _save_heatmap(speed_map, 'Forward speed (m/s)', 'exercise1_2_speed_heatmap.png')
    _save_heatmap(cot_map, 'CoT (J/m)', 'exercise1_2_cot_heatmap.png', log=True)
    _save_heatmap(ipl_map, 'Mean IPL_neur (rad)', 'exercise1_2_ipl_heatmap.png', cmap='coolwarm')


if __name__ == '__main__':
    exercise1_2(plot=True)

