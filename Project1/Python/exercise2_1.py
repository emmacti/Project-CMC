#!/usr/bin/env python3

import time
import os
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt


from farms_core import pylog
from farms_core.utils.profile import profile

from simulate import runsim
from cmc_controllers.metrics import LINKS_MASSES, filter_signals

BASE_PATH = 'logs/exercise2_1/'
PLOT_PATH = 'results'


def post_processing(base_path):
    """
    Q2.1 – Plot required signals for the first 5 seconds.

    Uses:
    - controller log (CPG phases/amplitudes + motor outputs)
    - sim sensors (joint angles, link positions)

    Produces:
    - oscillator states (theta, r)
    - muscle sum/difference (ML+MR, ML-MR)
    - body joint angles
    - CoM trajectory (XY)
    """
    # Load HDF5
    sim_result = base_path + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]

    # Load Controller
    with open(base_path + "controller.pkl", "rb") as f:
        controller_data = pickle.load(f)

    controller_state = controller_data["state"]  # (T, 3*n_osc)
    n_total = controller_state.shape[1]
    if n_total % 3 != 0:
        raise ValueError(
            f"Unexpected controller state width={n_total}, expected multiple of 3."
        )

    n_osc = n_total // 3
    n_body_joints = n_osc // 2
    if 2 * n_body_joints != n_osc:
        raise ValueError(
            f"Inconsistent oscillator count: n_body_joints={n_body_joints}, n_osc={n_osc}."
        )

    phases = controller_state[:, :n_osc]  # (T, n_osc)
    amplitudes = controller_state[:, n_osc:2*n_osc]  # (T, n_osc)

    # Motor outputs storage in CPGNetwork.state is the last 1/3 of the array.
    motor_storage = controller_state[:, 2*n_osc:3*n_osc]  # (T, n_osc)

    # In the CPG implementation, oscillator indices are interleaved:
    # even indices -> left muscle output, odd indices -> right muscle output.
    motor_left = motor_storage[:, 0::2]  # (T, n_body_joints)
    motor_right = motor_storage[:, 1::2]  # (T, n_body_joints)
    neural_sum = motor_left + motor_right
    neural_diff = motor_left - motor_right

    # Time window (first 5 seconds)
    t0 = float(sim_times[0])
    t_end = t0 + 5.0
    mask = sim_times <= t_end
    t = sim_times[mask]

    os.makedirs(PLOT_PATH, exist_ok=True)

    # 1) Oscillator states: theta and r (plot first 3 joints => 6 oscillators total)
    idx_joints_plot = min(3, n_body_joints)
    plt.figure(figsize=(10, 5))
    for j in range(idx_joints_plot):
        plt.plot(t, phases[mask, 2*j], label=f"theta left joint {j}")
        plt.plot(t, phases[mask, 2*j+1], linestyle="--", label=f"theta right joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("theta (rad)")
    plt.title("Oscillator phases (first 5s)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "exercise2_1_theta.png"), dpi=200)

    plt.figure(figsize=(10, 5))
    for j in range(idx_joints_plot):
        plt.plot(t, amplitudes[mask, 2*j], label=f"r left joint {j}")
        plt.plot(t, amplitudes[mask, 2*j+1], linestyle="--", label=f"r right joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("amplitude r")
    plt.title("Oscillator amplitudes (first 5s)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "exercise2_1_r.png"), dpi=200)

    # 2) Sum and difference of muscle outputs
    plt.figure(figsize=(10, 5))
    for j in range(idx_joints_plot):
        plt.plot(t, neural_sum[mask, j], label=f"ML+MR joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("ML+MR")
    plt.title("Muscle output sum (first 5s)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "exercise2_1_muscle_sum.png"), dpi=200)

    plt.figure(figsize=(10, 5))
    for j in range(idx_joints_plot):
        plt.plot(t, neural_diff[mask, j], label=f"ML-MR joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("ML-MR")
    plt.title("Muscle output difference (first 5s)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "exercise2_1_muscle_diff.png"), dpi=200)

    # 3) Body joint angles (plot all axial joints or up to 5)
    plt.figure(figsize=(10, 5))
    n_plot_joints = min(5, sensor_data_joints_positions.shape[1])
    for j in range(n_plot_joints):
        plt.plot(t, sensor_data_joints_positions[mask, j], label=f"joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("joint angle (rad)")
    plt.title("Body joint angles (first 5s)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "exercise2_1_joint_angles.png"), dpi=200)

    # 4) CoM trajectory in 2D
    n_links_total = sensor_data_links_positions.shape[1]
    masses = LINKS_MASSES[:n_links_total]
    mass_sum = float(masses.sum())
    if mass_sum <= 0:
        raise ValueError("Invalid total mass for CoM computation.")

    com_pos = (
        sensor_data_links_positions * masses[None, :, None]
    ).sum(axis=1) / mass_sum  # (T, 3)
    com_xy = com_pos[:, :2]

    plt.figure(figsize=(5, 5))
    plt.plot(com_xy[mask, 0], com_xy[mask, 1], linewidth=2)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.title("CoM trajectory in 2D (first 5s)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "exercise2_1_com_traj_xy.png"), dpi=200)
def main(**kwargs):
    """
    Q2.1 – Run the default CPG controller (Tab. params) and record a video.

    Runs with fixed drives d_left=d_right=3.
    """
    os.makedirs(PLOT_PATH, exist_ok=True)
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
            'PL': np.ones(7) * np.pi * 2 / 8,
            'coupling_weights_rostral': 5,
            'coupling_weights_caudal': 5,
            'coupling_weights_contra': 10,
            'init_phase': np.ascontiguousarray(np.random.default_rng(
                seed=42).uniform(
                0.0,
                2 * np.pi,
                size=16))}}

    tic = time.time()
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        recording='exercise2_1.mp4',
    )
    post_processing(BASE_PATH)
    pylog.info('Total simulation time: %s [s]', time.time() - tic)


def exercise2_1(**kwargs):
    """ex2.1 main"""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)
    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise2_1(plot=True)

