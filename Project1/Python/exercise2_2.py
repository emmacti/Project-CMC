#!/usr/bin/env python3
"""Run exercise 2.2 parameter sweeps and generate heatmaps/trajectory plots."""

import os
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    compute_frequency_amplitude_fft,
    compute_neural_phase_lags,
    compute_trajectory_curvature,
    filter_signals,
    LINKS_MASSES,
)

# Multiprocessing
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from simulate import run_multiple
MAX_WORKERS = 8  # adjust based on your hardware capabilities

# CPG parameters
BASE_PATH = 'logs/exercise2_2/'
PLOT_PATH = 'results'


def load_metrics_from_hdf5(hdf5_path):
    """Load speed and CoT metrics from an HDF5 simulation result."""
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    links_positions = sensor_data_links[:, :, 7:10]
    links_velocities = sensor_data_links[:, :, 14:17]
    joints_velocities = sensor_data_joints[:, :, 1]
    joints_torques = sensor_data_joints[:, :, 2]

    speed_forward, speed_lateral = compute_mechanical_speed(
        links_positions=links_positions,
        links_velocities=links_velocities,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=links_positions,
        joints_torques=joints_torques,
        joints_velocities=joints_velocities,
    )

    return speed_forward, speed_lateral, cot


def exercise2_2(**kwargs):
    # Parameter investigation (Question 2.2)
    plot = kwargs.pop('plot', False)

    n_body_joints = 8  # project 1 axial body joints
    pl_min = np.pi / (2.0 * n_body_joints)
    pl_max = 3.0 * np.pi / n_body_joints

    # Reuse defaults from exercise2_1
    rng = np.random.default_rng(seed=42)
    init_phase = rng.uniform(0.0, 2.0 * np.pi, size=16)

    base_controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': 3.0,
            'drive_right': 3.0,
            'd_low': 1.0,
            'd_high': 5.0,
            'a_rate': np.ones(n_body_joints) * 3.0,
            'offset_freq': np.ones(n_body_joints) * 1.0,
            'offset_amp': np.ones(n_body_joints) * 0.5,
            'G_freq': np.ones(n_body_joints) * 0.5,
            'G_amp': np.ones(n_body_joints) * 0.25,
            # Default phase bias per adjacent pair (from Tab.): PB = 2*pi/n_joint
            # For n_joint=8 this is pi/4. CPG controller accepts a scalar and expands it.
            'PL': float(2.0 * np.pi / n_body_joints),
            'coupling_weights_rostral': 5.0,
            'coupling_weights_caudal': 5.0,
            'coupling_weights_contra': 10.0,
            'init_phase': init_phase,
        }
    }

    # Ensure results directory exists
    os.makedirs(PLOT_PATH, exist_ok=True)

    def _token(x: float) -> str:
        # Must match simulate._as_filename_token for floats
        return f"{float(x):0.3f}"

    def _com_xy(links_positions: np.ndarray) -> np.ndarray:
        n_links_total = links_positions.shape[1]
        masses = LINKS_MASSES[:n_links_total]
        mass_sum = float(masses.sum())
        com_pos = (links_positions * masses[None, :, None]).sum(axis=1) / mass_sum
        return com_pos[:, :2]

    def _load_curvature_and_com(hdf5_path: str):
        with h5py.File(hdf5_path, "r") as f:
            sim_times = f['times'][:]
            sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]

        links_positions = sensor_data_links[:, :, 7:10]
        com_xy = _com_xy(links_positions)
        dt = float(sim_times[1] - sim_times[0])
        curvature = compute_trajectory_curvature(
            trajectory=com_xy,
            timestep=dt,
        )
        return curvature, com_xy

    def _heatmap(mat, title, xlabel, ylabel, fname, cmap='viridis', log=False):
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
            cmap=cmap,
            norm=norm,
        )
        plt.colorbar(im, label=title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, fname), dpi=200)

    # ------------------------------------------------------------------
    # Sweep 1 (Q2.2): d_left=d_right=d and PL varying
    # ------------------------------------------------------------------
    drive_values = np.linspace(2.0, 4.0, 5)
    pl_values = np.linspace(pl_min, pl_max, 5)

    # 2D grid: d vs PL
    parameter_grid_1 = {
        'drive': drive_values,
        'PL': pl_values,
    }

    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid_1,
        common_kwargs={'fast': True, 'headless': True},
    )

    forward_speed_map = np.zeros((len(drive_values), len(pl_values)), dtype=float)
    cot_map = np.zeros((len(drive_values), len(pl_values)), dtype=float)
    ipl_map = np.zeros((len(drive_values), len(pl_values)), dtype=float)
    phi_lock_map = np.zeros((len(drive_values), len(pl_values)), dtype=float)

    for i_d, d in enumerate(drive_values):
        for i_pl, pl in enumerate(pl_values):
            hdf5_path = os.path.join(
                BASE_PATH,
                f"simulation_drive{_token(d)}_PL{_token(pl)}.hdf5",
            )
            speed_fwd, _speed_lat, cot = load_metrics_from_hdf5(hdf5_path)
            forward_speed_map[i_d, i_pl] = speed_fwd
            cot_map[i_d, i_pl] = cot

            # Neural metrics (Q2.2): IPL_neur and phi_lock
            controller_pkl_path = os.path.join(
                BASE_PATH,
                f"controller_drive{_token(d)}_PL{_token(pl)}.pkl",
            )
            with open(controller_pkl_path, "rb") as f:
                controller_data = pickle.load(f)

            controller_state = controller_data["state"]  # (T, 3*n_osc)
            n_total = controller_state.shape[1]
            if n_total % 3 != 0:
                raise ValueError("Unexpected controller state shape in sweep 1.")
            n_osc = n_total // 3
            motor_storage = controller_state[:, 2*n_osc:3*n_osc]

            neural_left = motor_storage[:, 0::2]   # (T, n_body_joints)
            neural_right = motor_storage[:, 1::2]  # (T, n_body_joints)
            neural_signals = neural_left - neural_right

            # Skip transient for better FFT/phase-lag estimates
            with h5py.File(hdf5_path, "r") as f:
                sim_times = f['times'][:]
            skip = int(0.2 * len(sim_times))
            sim_times_ss = sim_times[skip:]
            neural_signals_ss = neural_signals[skip:]

            neural_signals_smoothed = filter_signals(
                times=sim_times_ss,
                signals=neural_signals_ss,
            )
            signal_freqs, _, _signal_amps = compute_frequency_amplitude_fft(
                times=sim_times_ss,
                smooth_signals=neural_signals_smoothed,
            )
            inds_couples = [
                [i, i + 1] for i in range(n_body_joints - 1)
            ]
            _, ipls_mean = compute_neural_phase_lags(
                times=sim_times_ss,
                smooth_signals=neural_signals_smoothed,
                freqs=signal_freqs,
                inds_couples=inds_couples,
            )
            ipl_map[i_d, i_pl] = float(ipls_mean)
            phi_lock_map[i_d, i_pl] = float(np.abs(ipls_mean - pl))

    # Heatmaps (x=PL, y=d)
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(
        forward_speed_map,
        origin='lower',
        aspect='auto',
        extent=[pl_values[0], pl_values[-1], drive_values[0], drive_values[-1]],
        cmap='viridis',
    )
    plt.colorbar(im, label='Forward speed (m/s)')
    plt.xlabel('Phase bias PL (rad)')
    plt.ylabel('Drive d (d_left=d_right)')
    plt.title('Forward speed heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'exercise2_2_forward_speed_heatmap.png'), dpi=200)

    plt.figure(figsize=(6, 4.5))
    show = np.clip(cot_map, 1e-9, None)
    norm = colors.LogNorm(vmin=np.min(show), vmax=np.max(show))
    im = plt.imshow(
        cot_map,
        origin='lower',
        aspect='auto',
        extent=[pl_values[0], pl_values[-1], drive_values[0], drive_values[-1]],
        cmap='magma',
        norm=norm,
    )
    plt.colorbar(im, label='CoT')
    plt.xlabel('Phase bias PL (rad)')
    plt.ylabel('Drive d (d_left=d_right)')
    plt.title('Cost of transport heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'exercise2_2_cot_heatmap.png'), dpi=200)

    # IPL_neur heatmap
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(
        ipl_map,
        origin='lower',
        aspect='auto',
        extent=[pl_values[0], pl_values[-1], drive_values[0], drive_values[-1]],
        cmap='coolwarm',
    )
    plt.colorbar(im, label='Mean IPL_neur (rad)')
    plt.xlabel('Phase bias PL (rad)')
    plt.ylabel('Drive d (d_left=d_right)')
    plt.title('Neural intersegmental phase lag (IPL_neur)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'exercise2_2_ipl_neur_heatmap.png'), dpi=200)

    # phi_lock heatmap (absolute mismatch to commanded PL)
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(
        phi_lock_map,
        origin='lower',
        aspect='auto',
        extent=[pl_values[0], pl_values[-1], drive_values[0], drive_values[-1]],
        cmap='viridis',
    )
    plt.colorbar(im, label='phi_lock = |IPL_neur - PL| (rad)')
    plt.xlabel('Phase bias PL (rad)')
    plt.ylabel('Drive d (d_left=d_right)')
    plt.title('Phase locking error (phi_lock)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'exercise2_2_phi_lock_heatmap.png'), dpi=200)

    # ------------------------------------------------------------------
    # Sweep 2 (Q2.2): differential drive d_left vs d_right
    # ------------------------------------------------------------------
    parameter_grid_2 = {
        'drive_left': drive_values,
        'drive_right': drive_values,
    }

    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid_2,
        common_kwargs={'fast': True, 'headless': True},
    )

    curvature_map = np.zeros((len(drive_values), len(drive_values)), dtype=float)

    # Heatmap expects matrix index [i_left, i_right]
    for i_left, d_left in enumerate(drive_values):
        for i_right, d_right in enumerate(drive_values):
            hdf5_path = os.path.join(
                BASE_PATH,
                f"simulation_drive_left{_token(d_left)}_drive_right{_token(d_right)}.hdf5",
            )
            curvature, _com_xy = _load_curvature_and_com(hdf5_path)
            curvature_map[i_left, i_right] = curvature

    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(
        curvature_map,
        origin='lower',
        aspect='auto',
        extent=[drive_values[0], drive_values[-1], drive_values[0], drive_values[-1]],
        cmap='coolwarm',
    )
    plt.colorbar(im, label='Mean curvature')
    plt.xlabel('Drive d_right (rad/s)')
    plt.ylabel('Drive d_left')
    plt.title('Curvature heatmap (d_left vs d_right)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'exercise2_2_curvature_heatmap.png'), dpi=200)

    # Example trajectories: corners of the grid
    corner_pairs = [
        (drive_values[0], drive_values[0]),
        (drive_values[0], drive_values[-1]),
        (drive_values[-1], drive_values[0]),
        (drive_values[-1], drive_values[-1]),
    ]
    for d_left, d_right in corner_pairs:
        _, com_xy = _load_curvature_and_com(
            os.path.join(
                BASE_PATH,
                f"simulation_drive_left{_token(d_left)}_drive_right{_token(d_right)}.hdf5",
            )
        )
        plt.figure(figsize=(5, 5))
        plt.plot(com_xy[:, 0], com_xy[:, 1], linewidth=2)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.title(f'CoM trajectory (d_left={d_left:.2f}, d_right={d_right:.2f})')
        plt.tight_layout()
        out = f"exercise2_2_com_traj_dL{_token(d_left)}_dR{_token(d_right)}.png"
        plt.savefig(os.path.join(PLOT_PATH, out), dpi=200)

    if plot:
        plt.show()


if __name__ == '__main__':
    exercise2_2(plot=True)

