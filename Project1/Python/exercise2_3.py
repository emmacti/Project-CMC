#!/usr/bin/env python3

import os
import h5py
import matplotlib
matplotlib.use('Agg') # I added this
import matplotlib.pyplot as plt


from farms_core import pylog

from cmc_controllers.metrics import *
from simulate import runsim


BASE_PATH = 'logs/exercise2_3/'
PLOT_PATH = 'results'
ANIMAL_DATA_PATH = 'cmc_project_pack/models/a2sw5_cycle_smoothed.csv'

def get_animal_data(path: str):
    """Load and compute neural metrics from provided animal joint angles."""
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    times = data[:, 0]
    joint_angles_deg = data[:, 1:9]  # 8 axial joints
    joint_angles = np.deg2rad(joint_angles_deg)

    smooth = filter_signals(times=times, signals=joint_angles)
    freqs, _, amps = compute_frequency_amplitude_fft(times=times, smooth_signals=smooth)

    # IPL (between adjacent joints)
    inds = [[i, i + 1] for i in range(7)]
    _, ipls = compute_neural_phase_lags(
        times=times,
        smooth_signals=smooth,
        freqs=freqs,
        inds_couples=inds,
    )
    return times, joint_angles, freqs, amps, ipls


def _controller_metrics_from_logs(hdf5_path: str):
    """Compute speed/CoT from sim logs and neural metrics from joint angles."""
    with h5py.File(hdf5_path, "r") as f:
        times = f['times'][:]
        sensor_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    links_positions = sensor_links[:, :, 7:10]
    links_velocities = sensor_links[:, :, 14:17]
    joints_positions = sensor_joints[:, :, 0][:, :8]  # first 8 = body yaw joints
    joints_velocities = sensor_joints[:, :, 1]
    joints_torques = sensor_joints[:, :, 2]

    # Mechanical performance
    speed_fwd, speed_lat = compute_mechanical_speed(
        links_positions=links_positions,
        links_velocities=links_velocities,
    )
    _energy, cot = compute_mechanical_energy_and_cot(
        times=times,
        links_positions=links_positions,
        joints_torques=joints_torques,
        joints_velocities=joints_velocities,
    )

    # Neural-ish metrics from joint kinematics (consistent with animal angles)
    smooth = filter_signals(times=times, signals=joints_positions)
    freqs, _, amps = compute_frequency_amplitude_fft(times=times, smooth_signals=smooth)
    inds = [[i, i + 1] for i in range(7)]
    _ipls, ipls_mean = compute_neural_phase_lags(
        times=times,
        smooth_signals=smooth,
        freqs=freqs,
        inds_couples=inds,
    )
    return {
        "times": times,
        "joint_angles": joints_positions,
        "freqs": freqs,
        "amps": amps,
        "ipls_mean": float(ipls_mean),
        "speed_fwd": float(speed_fwd),
        "speed_lat": float(speed_lat),
        "cot": float(cot),
    }

def exercise2_3(**kwargs):
    """
    Q2.3 – Compare provided animal swimming kinematics to the controller.

    Steps:
    - Run one CPG controller simulation (baseline parameters).
    - Compute controller joint-angle metrics (freq/amp/IPL) and mechanical metrics (speed/CoT).
    - Load animal axial joint angles from `cmc_project_pack/models/a2sw5_cycle_smoothed.csv`
      and compute the same joint-angle metrics (freq/amp/IPL).
    - Save comparison plots to `results/`.
    """
    pylog.set_level('warning')
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(PLOT_PATH, exist_ok=True)

    # Run one controller simulation 
    controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': 3,
            'drive_right': 3,
            'd_low': 1,
            'd_high': 5,
            'a_rate': np.ones(8) * 3,
            'offset_freq': np.ones(8) * 1,
            'offset_amp': np.ones(8) * 0.5,
            'G_freq': np.ones(8) * 0.5,
            'G_amp': np.ones(8) * 0.25,
            'PL': float(2*np.pi/8),
            'coupling_weights_rostral': 5,
            'coupling_weights_caudal': 5,
            'coupling_weights_contra': 10,
            'init_phase': np.random.default_rng(seed=42).uniform(0.0, 2*np.pi, 16),
        }
    }

    controller_tuned = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': 3,
            'drive_right': 3,
            'd_low': 1,
            'd_high': 5,
            'a_rate': np.ones(8) * 3,
            'offset_freq': np.ones(8) * 1,
            'offset_amp': np.ones(8) * 0.5,
            'G_freq': np.ones(8) * 0.5,
            'G_amp': np.ones(8) * 0.25,
            'PL': float(2*np.pi/8),
            'coupling_weights_rostral': 5,
            'coupling_weights_caudal': 5,
            'coupling_weights_contra': 10,
            'init_phase': np.random.default_rng(seed=42).uniform(0.0, 2*np.pi, 16),
        }
    }


    runsim(
        controller=controller,
        base_path=BASE_PATH,
        headless=kwargs.pop('headless', True),
        fast=kwargs.pop('fast', True),
        runtime_n_iterations=kwargs.pop('runtime_n_iterations', 5001),
        runtime_buffer_size=kwargs.pop('runtime_buffer_size', 5001),
        hdf5_name='simulation.hdf5',
        controller_name='controller.pkl',
    )

    ctrl = _controller_metrics_from_logs(os.path.join(BASE_PATH, "simulation.hdf5"))
    a_times, a_angles, a_freqs, a_amps, a_ipls = get_animal_data(ANIMAL_DATA_PATH)

    # Plots: frequency/amplitude per joint
    plt.figure(figsize=(7, 4))
    plt.plot(a_freqs, 'o-', label='animal')
    plt.plot(ctrl["freqs"], 's--', label='controller (joint angles)')
    plt.title("Q2.3 Frequency comparison")
    plt.xlabel("Joint index")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "exercise2_3_freq.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(a_amps, 'o-', label='animal')
    plt.plot(ctrl["amps"], 's--', label='controller (joint angles)')
    plt.title("Q2.3 Amplitude comparison")
    plt.xlabel("Joint index")
    plt.ylabel("Amplitude (rad)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "exercise2_3_amp.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(1, a_ipls, label='animal IPL (adjacent)')
    plt.bar(2,ctrl["ipls_mean"], color='tab:orange', label='controller mean IPL')
    plt.title("Q2.3 IPL comparison")
    plt.xlabel("Segment pair index (i,i+1)")
    plt.ylabel("Phase lag (rad)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "exercise2_3_ipl.png"), dpi=200)
    plt.close()

    # Print key locomotion performance from controller run
    print("Controller mechanical metrics:")
    print("  speed_fwd:", ctrl["speed_fwd"])
    print("  speed_lat:", ctrl["speed_lat"])
    print("  CoT:", ctrl["cot"])


if __name__ == '__main__':
    exercise2_3(plot=True)