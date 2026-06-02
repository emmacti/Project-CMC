"""Exercise 3: Limb and Spine Coordination while walking"""

import os
import numpy as np
from salamandra_simulation.simulation import simulation, simulation_sweep
from simulation_parameters import SimulationParameters
from farms_core import pylog
import matplotlib.pyplot as plt

from plot_results import (
    load_data,
    compute_speed,
)

def _links_velocity_from_positions(links_positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute link velocities from positions with finite differences.

    links_positions shape: (T, n_links, 3)
    returns shape: (T, n_links, 3)
    """
    if links_positions.shape[0] < 2:
        return np.zeros_like(links_positions)
    vel = np.zeros_like(links_positions, dtype=float)
    vel[1:] = np.diff(links_positions, axis=0) / float(dt)
    vel[0] = vel[1]
    return vel


def _compute_cot(exp_data, nsteps_considered=400) -> float:
    """Compute CoT consistent with Project 1.

    Project 1 defines energy as integral of *positive* mechanical power only:
      P_j(t) = max(tau_j(t) * qdot_j(t), 0)
      E = dt * sum_t sum_j P_j(t)
    And cost of transport:
      CoT = E / D_fwd
    """
    data = exp_data.animats[0]
    dt = float(exp_data.timestep)
    joints_vel = np.asarray(data.sensors.joints.velocities_all())
    joints_tau = np.asarray(data.sensors.joints.motor_torques_all())
    power = np.maximum(joints_tau * joints_vel, 0.0)
    energy = float(np.sum(power[-nsteps_considered:]) * dt)

    links_positions = np.asarray(data.sensors.links.urdf_positions())
    links_vel = (
        np.asarray(data.sensors.links.urdf_velocities())
        if hasattr(data.sensors.links, 'urdf_velocities')
        else _links_velocity_from_positions(links_positions, dt=dt)
    )
    speed_forward, _ = compute_speed(
        links_positions=links_positions,
        links_vel=links_vel,
        nsteps_considered=nsteps_considered,
    )
    distance = max(1e-6, float(speed_forward * (nsteps_considered * dt)))
    return float(energy / distance)


def exercise_3_disable_limb_spine_coupling(timestep):
    """ Walk with disabled limb-spine limbs """
    os.makedirs('./logs/ex3_disable_coupling/', exist_ok=True)
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        drive=2.5,
        disable_limb_spine_coupling=True,
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=False,
        record=True,
        output='logs/ex3_disable_coupling/sim_0',
        record_path='logs/ex3_disable_coupling/disable_limb_spine.mp4',
        verbose=False,
    )
    return


def exercise_3_limb_spine_antiphase(timestep):
    """ Walk with limb-spine in anti-phase """
    os.makedirs('./logs/ex3_antiphase/', exist_ok=True)
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        drive=2.5,
        limb_spine_phase_offset=np.pi,
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=False,
        record=True,
        output='logs/ex3_antiphase/sim_0',
        record_path='logs/ex3_antiphase/antiphase.mp4',
        verbose=False,
    )
    return


def exercise_3a_coordination(timestep):
    """Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and limb oscillators.

    """
    os.makedirs('./logs/ex3a/', exist_ok=True)
    drives = np.linspace(1.5, 3.2, 5)
    phase_offsets = np.linspace(-np.pi, np.pi, 9)
    parameter_set = [
        SimulationParameters(
            duration=10,
            timestep=timestep,
            drive=float(d),
            limb_spine_phase_offset=float(phi),
        )
        for d in drives
        for phi in phase_offsets
    ]

    sim_args = [
        {
            'sim_parameters': sim_parameters,
            'arena': 'land',
            'fast': True,
            'headless': True,
            'output': f'logs/ex3a/sim_{simulation_i}',
            'verbose': False,
        }
        for simulation_i, sim_parameters in enumerate(parameter_set)
    ]
    simulation_sweep(sim_args, processes=4)

    results_speed = []
    results_cot = []
    for simulation_i, sim_parameters in enumerate(parameter_set):
        exp_data, _ = load_data("logs/ex3a/sim_{}", simulation_i)
        data = exp_data.animats[0]
        dt = float(exp_data.timestep)
        links_positions = np.asarray(data.sensors.links.urdf_positions())
        links_vel = (
            np.asarray(data.sensors.links.urdf_velocities())
            if hasattr(data.sensors.links, 'urdf_velocities')
            else _links_velocity_from_positions(links_positions, dt=dt)
        )
        speed_forward, _ = compute_speed(links_positions, links_vel)
        cot = _compute_cot(exp_data)
        results_speed.append([sim_parameters.limb_spine_phase_offset, sim_parameters.drive, speed_forward])
        results_cot.append([sim_parameters.limb_spine_phase_offset, sim_parameters.drive, cot])

    results_speed = np.asarray(results_speed, dtype=float)
    results_cot = np.asarray(results_cot, dtype=float)
    np.save('logs/ex3a/results_speed.npy', results_speed)
    np.save('logs/ex3a/results_cot.npy', results_cot)

    plt.figure('Ex3a speed')
    from plot_results import plot_2d
    plot_2d(results_speed, labels=['Phase offset [rad]', 'Drive', 'Forward speed [m/s]'])
    plt.figure('Ex3a CoT')
    plot_2d(results_cot, labels=['Phase offset [rad]', 'Drive', 'CoT'])
    plt.savefig('logs/ex3a/ex3a_plots.png', dpi=150)
    # # For sweeps with many simulations running in parallel
    # parameter_set = [
    #     SimulationParameters(...)
    #     for ... in ...
    #     for ... in ...
    # ]
    # os.makedirs('./logs/sweep_3a/', exist_ok=True)
    # simulation_sweep([
    #     {
    #         'sim_parameters': sim_parameters,
    #         'arena': 'land',
    #         'fast': True,  # For fast mode (not real-time)
    #         'headless': True,  # For headless mode (No GUI, could be faster)
    #         'output': f'logs/ex3a/simulation_{simulation_i}',
    #         'verbose': False,
    #     }
    #     for simulation_i, sim_parameters in enumerate(parameter_set)
    # ], processes=4)  # Adjust based on your hardware
    return


def exercise_3b_coordination(timestep):
    """Exercise 3b Limb and Spine coordination

    This exercise explores how axial and limb amplitudes affect coordination.

    Run the simulations for different axial and limb amplitudes.

    """
    # Use exercise_example.py for reference
    os.makedirs('./logs/ex3b/', exist_ok=True)
    drives = 2.5

    # Use the optimal phase offset from exercise 3a if available
    best_phi = 0.0
    try:
        results_speed = np.load('logs/ex3a/results_speed.npy')
        # results_speed columns: [phi, drive, speed]
        best_phi = float(results_speed[np.argmax(results_speed[:, 2]), 0])
        pylog.info('Ex3b using best phase offset from Ex3a: %.3f rad', best_phi)
    except Exception:
        pylog.warning('Ex3a results not found; using phase offset = 0 rad')
    axial_gains = np.linspace(0.0, 2.0, 9)
    limb_gains = np.linspace(0.0, 2.0, 9)

    parameter_set = [
        SimulationParameters(
            duration=10,
            timestep=timestep,
            drive=float(drives),
            limb_spine_phase_offset=best_phi,
            axial_amp_gain=float(ga),
            limb_amp_gain=float(gl),
        )
        for ga in axial_gains
        for gl in limb_gains
    ]
    sim_args = [
        {
            'sim_parameters': sim_parameters,
            'arena': 'land',
            'fast': True,
            'headless': True,
            'output': f'logs/ex3b/sim_{simulation_i}',
            'verbose': False,
        }
        for simulation_i, sim_parameters in enumerate(parameter_set)
    ]
    simulation_sweep(sim_args, processes=4)

    results_speed = []
    results_cot = []
    for simulation_i, sim_parameters in enumerate(parameter_set):
        exp_data, _ = load_data("logs/ex3b/sim_{}", simulation_i)
        data = exp_data.animats[0]
        dt = float(exp_data.timestep)
        links_positions = np.asarray(data.sensors.links.urdf_positions())
        links_vel = (
            np.asarray(data.sensors.links.urdf_velocities())
            if hasattr(data.sensors.links, 'urdf_velocities')
            else _links_velocity_from_positions(links_positions, dt=dt)
        )
        speed_forward, _ = compute_speed(links_positions, links_vel)
        cot = _compute_cot(exp_data)
        results_speed.append([sim_parameters.axial_amp_gain, sim_parameters.limb_amp_gain, speed_forward])
        results_cot.append([sim_parameters.axial_amp_gain, sim_parameters.limb_amp_gain, cot])

    results_speed = np.asarray(results_speed, dtype=float)
    results_cot = np.asarray(results_cot, dtype=float)
    np.save('logs/ex3b/results_speed.npy', results_speed)
    np.save('logs/ex3b/results_cot.npy', results_cot)

    from plot_results import plot_2d
    plt.figure('Ex3b speed')
    plot_2d(results_speed, labels=['Axial gain', 'Limb gain', 'Forward speed [m/s]'])
    plt.figure('Ex3b CoT')
    plot_2d(results_cot, labels=['Axial gain', 'Limb gain', 'CoT'])
    plt.savefig('logs/ex3b/ex3b_plots.png', dpi=150)
    return


if __name__ == '__main__':
    exercise_3_disable_limb_spine_coupling(timestep=5e-3)
    exercise_3_limb_spine_antiphase(timestep=5e-3)
    exercise_3a_coordination(timestep=5e-3)
    exercise_3b_coordination(timestep=5e-3)

