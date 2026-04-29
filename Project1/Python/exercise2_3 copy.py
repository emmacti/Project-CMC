#!/usr/bin/env python3

import os
import numpy as np
from uuid import uuid4
from datetime import datetime
import h5py
import matplotlib
matplotlib.use('Agg') # I added this
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


from farms_core import pylog

from cmc_controllers.metrics import *
from simulate import runsim


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, 'logs', 'exercise2_3')
PLOT_PATH = os.path.join(SCRIPT_DIR, 'results')
ANIMAL_DATA_PATH = os.path.join(SCRIPT_DIR, 'cmc_project_pack', 'models', 'a2sw5_cycle_smoothed.csv')
DEFAULT_X = np.array([
    2.711e+00,
    5.736e+00,
    2.483e+00,
    8.103e+00,
    4.714e+00,
    3.423e+00,
    4.510e+00,
    9.309e+00,
    1.024e+00,
    6.415e+00,
    9.453e+00,
    1.890e+00,
], dtype=float)
DEFAULT_X = np.array([
    4.537e+00,
    6.061e+00,
    2.141e+00,
    8.022e+00,
    2.946e+00,
    3.037e-01,
    8.304e-01,
    1.619e+00,
    9.834e-01,
    4.333e+00,
    5.731e+00,
    3.733e-02
], dtype=float)

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
        "n_steps": int(len(times)),
        "joint_angles": joints_positions,
        "freqs": freqs,
        "amps": amps,
        "ipls_mean": float(ipls_mean),
        "speed_fwd": float(speed_fwd),
        "speed_lat": float(speed_lat),
        "cot": float(cot),
    }


def _controller_from_x(x):
    return {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': float(x[0]),
            'drive_right': float(x[1]),
            'd_low': float(x[2]),
            'd_high': float(x[3]),
            'a_rate': np.ones(8) * float(x[4]),
            'offset_freq': np.ones(8) * float(x[5]),
            'offset_amp': np.ones(8) * float(x[6]),
            'G_freq': np.ones(8) * float(x[7]),
            'G_amp': np.ones(8) * float(x[8]),
            'PL': float(2*np.pi/8),
            'coupling_weights_rostral': float(x[9]),
            'coupling_weights_caudal': float(x[10]),
            'coupling_weights_contra': float(x[11]),
            'init_phase': np.random.default_rng(seed=42).uniform(0.0, 2*np.pi, 16),
        }
    }


def _plot_comparison(ctrl, animal_freqs, animal_amps, animal_ipls, plot_prefix='exercise2_3'):
    plt.figure(figsize=(7, 4))
    plt.plot(animal_freqs, 'o-', label='animal')
    plt.plot(ctrl['freqs'], 's--', label='controller (joint angles)')
    plt.title('Q2.3 Frequency comparison')
    plt.xlabel('Joint index')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f'{plot_prefix}_freq.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(animal_amps, 'o-', label='animal')
    plt.plot(ctrl['amps'], 's--', label='controller (joint angles)')
    plt.title('Q2.3 Amplitude comparison')
    plt.xlabel('Joint index')
    plt.ylabel('Amplitude (rad)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f'{plot_prefix}_amp.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(0, animal_ipls, label='animal IPL (adjacent)')
    plt.bar(1,ctrl['ipls_mean'], color='tab:orange', label='controller mean IPL')
    plt.title('Q2.3 IPL comparison')
    plt.xlabel('Segment pair index (i,i+1)')
    plt.ylabel('Phase lag (rad)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f'{plot_prefix}_ipl.png'), dpi=200)
    plt.close()


def _run_single_simulation(
    x,
    *,
    headless=True,
    fast=True,
    runtime_n_iterations=5001,
    runtime_buffer_size=5001,
    tag='final',
    verbose=True,
):
    controller = _controller_from_x(x)
    hdf5_name = f'simulation_tuned_{tag}.hdf5'
    controller_name = f'controller_tuned_{tag}.pkl'
    hdf5_path = os.path.join(BASE_PATH, hdf5_name)
    controller_path = os.path.join(BASE_PATH, controller_name)

    try:
        if verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting simulation: {tag}")
        runsim(
            controller=controller,
            base_path=BASE_PATH,
            headless=headless,
            fast=fast,
            runtime_n_iterations=runtime_n_iterations,
            runtime_buffer_size=runtime_buffer_size,
            hdf5_name=hdf5_name,
            controller_name=controller_name,
        )
        ctrl = _controller_metrics_from_logs(hdf5_path)
        if verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished simulation: {tag}")
    finally:
        for p in (hdf5_path, controller_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    return ctrl

def exercise2_3_tuned(opti: bool = True, **kwargs):
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

    # Precompute animal metrics and scaling factors for normalization.
    # Use representative magnitudes with a floor so tiny variance does not explode the loss.
    animal_times, animal_angles, animal_freqs, animal_amps, animal_ipls = get_animal_data(ANIMAL_DATA_PATH)
    animal_ipl_target = float(np.mean(animal_ipls))
    eps = 1e-6
    freq_scale = max(float(np.mean(np.abs(animal_freqs))), float(np.ptp(animal_freqs)), 1.0)
    amp_scale = max(float(np.mean(np.abs(animal_amps))), float(np.ptp(animal_amps)), 0.1)
    ipl_scale = max(float(np.mean(np.abs(animal_ipls))), float(np.ptp(animal_ipls)), 0.1)
    eval_counter = {'value': 0}
    verbose = bool(kwargs.get('verbose', True))

    def loss(x):
        w_f = 1.0
        w_a = 1.0
        w_p = 1.0
        # use unique filenames per evaluation to avoid file reuse/corruption
        eval_counter['value'] += 1
        eval_tag = f"eval_{eval_counter['value']:05d}_{uuid4().hex[:8]}"
        try:
            ctrl = _run_single_simulation(
                x,
                headless=kwargs.get('headless', True),
                fast=kwargs.get('fast', True),
                runtime_n_iterations=kwargs.get('runtime_n_iterations', 5001),
                runtime_buffer_size=kwargs.get('runtime_buffer_size', 5001),
                tag=eval_tag,
                verbose=verbose,
            )
        except Exception:
            return 1e12

        # reject bad simulations early so the optimizer does not chase NaNs
        if not (
            np.all(np.isfinite(ctrl['freqs'])) and
            np.all(np.isfinite(ctrl['amps'])) and
            np.isfinite(ctrl['ipls_mean'])
        ):
            return 1e12

        expected_steps = int(kwargs.get('runtime_n_iterations', 5001))
        if ctrl['n_steps'] < max(20, int(0.9 * expected_steps)):
            return 1e12

        # normalized RMSE terms, all dimensionless and roughly comparable
        freq_err = np.sqrt(np.mean((ctrl['freqs'] - animal_freqs)**2)) / freq_scale
        amp_err = np.sqrt(np.mean((ctrl['amps'] - animal_amps)**2)) / amp_scale
        ipl_err = abs(ctrl['ipls_mean'] - animal_ipl_target) / ipl_scale
        loss_val = w_f * freq_err + w_a * amp_err + w_p * ipl_err
        if verbose:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            max_evals = (int(kwargs.get('opt_maxiter', 10)) + 1) * int(kwargs.get('opt_popsize', 5)) * len(bounds)
            print(f"[{now}] Eval {eval_counter['value']}/{max_evals} Loss: {loss_val:.4f} "
                  f"(freq_err: {freq_err:.4f}, amp_err: {amp_err:.4f}, ipl_err: {ipl_err:.4f})")

        return loss_val

    def optimizer_callback(xk, convergence):
        if verbose:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            generation = optimizer_callback.generation + 1
            print(f"[{now}] Generation {generation}/{optimizer_callback.maxiter} "
                  f"convergence={convergence:.6f}")
        optimizer_callback.generation += 1
        return False

    optimizer_callback.generation = 0
    optimizer_callback.maxiter = int(kwargs.get('opt_maxiter', 10))
    
    
    bounds = [
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
    ]
    if opti:
        # Derivative-free global optimizer with limited evaluations for testing
        maxiter = optimizer_callback.maxiter
        popsize = int(kwargs.get('opt_popsize', 5))
        polish = bool(kwargs.get('polish', True))
        result = differential_evolution(
            loss,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            polish=polish,
            updating='deferred',
            callback=optimizer_callback,
        )
        best_x = result.x
        print("Optimization result:", result)
    else:
        best_x = DEFAULT_X
        print("Skipping optimizer, using fixed x:", best_x)

    ctrl = _run_single_simulation(
        best_x,
        headless=kwargs.get('headless', True),
        fast=kwargs.get('fast', True),
        runtime_n_iterations=kwargs.get('runtime_n_iterations', 5001),
        runtime_buffer_size=kwargs.get('runtime_buffer_size', 5001),
        tag='final',
        verbose=verbose,
    )

    _plot_comparison(ctrl, animal_freqs, animal_amps, animal_ipls)

    print("Controller mechanical metrics:")
    print("  speed_fwd:", ctrl["speed_fwd"])
    print("  speed_lat:", ctrl["speed_lat"])
    print("  CoT:", ctrl["cot"])

if __name__ == '__main__':
    #exercise2_3_tuned(opti=False, runtime_n_iterations=200, plot=False)
    exercise2_3_tuned(opti=False, verbose = True)
    
