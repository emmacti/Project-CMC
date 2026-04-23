#!/usr/bin/env python3

import os
import pickle
import h5py
import matplotlib.pyplot as plt

from farms_core.utils.profile import profile
from farms_core import pylog

from simulate import runsim
from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_frequency_amplitude_fft,
    compute_mechanical_speed,
    compute_neural_phase_lags,
    filter_signals,
)
from cmc_controllers.metrics import LINKS_MASSES

BASE_PATH = 'logs/exercise1_1/'
PLOT_PATH = 'results'


def post_processing():
    """
    Q1.2 – Metrics + plots on the WaveController run.

    Loads `logs/exercise1_1/simulation.hdf5` and `controller.pkl`, computes:
    - NM1: neural frequency + amplitude (FFT)
    - NM2: intersegmental phase lag (IPL_neur)
    - MM2: forward/lateral speed
    - MM4: energy + CoT
    Then saves:
    - joint-angle time series (few cycles)
    - CoM trajectory (XY)
    """
    # Load HDF5
    sim_result = BASE_PATH + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_links_velocities = sensor_data_links[:, :, 14:17]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]
    sensor_data_joints_velocities = sensor_data_joints[:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[:, :, 2]

    # Load Controller
    with open(BASE_PATH + "controller.pkl", "rb") as f:
        controller_data = pickle.load(f)

    indices = controller_data["indices"]
    neural_signals = (
        controller_data["state"][:, indices['left_body_idx']]
        - controller_data["state"][:, indices['right_body_idx']]
    )
    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)

    # Metrics computation
    freq, _, amp = compute_frequency_amplitude_fft(
        times=sim_times, smooth_signals=neural_signals_smoothed)

    inds_couples = [[i, i+1]
                    for i in range(neural_signals_smoothed.shape[1] - 1)]
    _, ipls_mean = compute_neural_phase_lags(times=sim_times,
                                             smooth_signals=neural_signals_smoothed,
                                             freqs=freq,
                                             inds_couples=inds_couples)

    mech_freq, mech_amp = compute_mechanical_frequency_amplitude_fft(
        times=sim_times,
        signals=sensor_data_joints_positions[:, :8],
    )

    speed_forward, speed_lateral = compute_mechanical_speed(
        links_positions=sensor_data_links_positions,
        links_velocities=sensor_data_links_velocities,
    )

    energy, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=sensor_data_links_positions,
        joints_torques=sensor_data_joints_torques,
        joints_velocities=sensor_data_joints_velocities,
    )

    print('Estimated neural metrics:')
    print('Frequencies: ', freq, '\nAmplitudes: ', amp,
          '\nMean phase lags (radians): ', ipls_mean)
    print('Estimated mechanical metrics:')
    print(
        'Frequencies: ',
        mech_freq,
        '\nAmplitudes: ',
        mech_amp,
        '\nforward speed: ',
        speed_forward,
        '\nlateral speed: ',
        speed_lateral,
        '\nEnergy: ',
        energy,
        '\nCoT: ',
        cot)

    # Plots: joint angles + CoM trajectory in 2D
    os.makedirs(PLOT_PATH, exist_ok=True)

    # Joint angles: show a few axial joints (first 8)
    n_plot_joints = min(8, sensor_data_joints_positions.shape[1])
    plt.figure(figsize=(9, 4))
    for j in range(n_plot_joints):
        plt.plot(sim_times, sensor_data_joints_positions[:, j], label=f'joint {j}')
    plt.xlabel('time (s)')
    plt.ylabel('joint angle (rad)')
    plt.title('Axial joint angles (first 8)')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'exercise1_1_joint_angles.png'), dpi=200)

    # CoM trajectory in XY
    n_links_total = sensor_data_links_positions.shape[1]
    masses = LINKS_MASSES[:n_links_total]
    mass_sum = float(masses.sum())
    com_pos = (sensor_data_links_positions * masses[None, :, None]).sum(axis=1) / mass_sum

    plt.figure(figsize=(5, 5))
    plt.plot(com_pos[:, 0], com_pos[:, 1], linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.title('CoM trajectory (XY)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'exercise1_1_com_trajectory_xy.png'), dpi=200)


def main(**kwargs):
    """
    Q1.1 – Run WaveController with A=2, f=2Hz, TWL=1 and record a video.

    Output logs go to `logs/exercise1_1/`.
    """
    os.makedirs(PLOT_PATH, exist_ok=True)
    controller = {
        'loader': 'cmc_controllers.wave_controller.WaveController',
        'config': {'freq': 2.0,
                   'twl': 1.0,
                   'amp': 2.0}
    }

    runsim(
        controller=controller,
        base_path=BASE_PATH,
        recording='exercise1_1.mp4',
    )

    post_processing()


def exercise1_1(**kwargs):
    """Entry point for exercise 1.1 with optional plotting and runtime flags."""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise1_1(plot=True)

