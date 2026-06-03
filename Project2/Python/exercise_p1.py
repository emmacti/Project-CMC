"""[Project1] Exercise 1: Implement & run network without MuJoCo"""

import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork


@dataclass
class DataState:
    state: SalamandraState


def run_network(duration, update=False, drive=0, timestep=1e-2):
    """ Run network without MuJoCo and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        True: use the prescribed drive parameter, False: update the drive during the simulation
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)

    # Accept scalar drive (constant) or an array (time-varying)
    if np.isscalar(drive):
        drive_vector = np.full(n_iterations, float(drive), dtype=float)
    else:
        drive_vector = np.asarray(drive, dtype=float)
        assert len(drive_vector) == n_iterations, (
            "drive array must have length n_iterations"
        )

    sim_parameters = SimulationParameters(
        drive=float(drive_vector[0]),
        amplitude_gradient=None,
        phase_lag_body=None,
        # Feel free to include parameters
    )
    if np.isscalar(drive):
        pylog.info(
            'Drive is scalar. For a ramp experiment, pass a vector of length n_iterations.'
        )
    state = SalamandraState.salamandra_robot(n_iterations, n_oscillators=32)
    network = SalamandraNetwork(
        sim_parameters,
        n_iterations,
        DataState(
            state=state))
    osc_left = np.arange(0, 16, 2)
    osc_right = np.arange(1, 16, 2)
    osc_legs = np.arange(16, 32)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs

    drive_log = np.zeros(n_iterations, dtype=float)
    drive_log[0] = drive_vector[0]

    nominal_amp_log = np.zeros(
        [n_iterations, network.robot_parameters.n_oscillators],
        dtype=float,
    )
    nominal_amp_log[0, :] = network.robot_parameters.nominal_amplitudes

    # Run network ODE and log data
    pylog.info('Running network ODE (no MuJoCo)')
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        # Keep backwards compatibility with the "update" flag:
        # - update=False: still updates the drive if a drive vector is provided
        # - update=True: explicitly updates each step
        current_drive = float(drive_vector[i + 1])
        if update or (not np.isscalar(drive)):
            network.robot_parameters.update(
                SimulationParameters(drive=current_drive)
            )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
        drive_log[i+1] = current_drive
        nominal_amp_log[i+1, :] = network.robot_parameters.nominal_amplitudes
    toc = time.time()

    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    motor = network.get_motor_activations(iteration=None)

    plt.figure('Oscillator phases')
    for k in range(phases_log.shape[1]):
        plt.plot(times, np.unwrap(phases_log[:, k]), linewidth=0.8)
    plt.xlabel('Time [s]')
    plt.ylabel('Unwrapped phase [rad]')
    plt.grid(True)

    plt.figure('Oscillator amplitudes')
    for k in range(amplitudes_log.shape[1]):
        plt.plot(times, amplitudes_log[:, k], linewidth=0.8)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude r')
    plt.grid(True)

    plt.figure('Motor outputs (first 8 joints)')
    for j in range(min(motor.shape[1], 8)):
        plt.plot(times, motor[:, j], label=f'joint{j}')
    plt.xlabel('Time [s]')
    plt.ylabel('Motor command')
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True)

    # Richer "Project2_old" style plots (kept here so students can compare)
    body_output = amplitudes_log[:, :16] * (1 + np.cos(phases_log[:, :16]))
    limb_output = amplitudes_log[:, 16:] * (1 + np.cos(phases_log[:, 16:]))

    fig1, axes1 = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True,
        num='Body oscillator output (x_i)',
    )
    ax = axes1[0]
    for j in range(0, 16, 2):
        ax.plot(times, body_output[:, j], label=f'x{j}', alpha=0.8)
    ax.set_ylabel('x Body (left)')
    ax.set_title('Body oscillator motor outputs')
    ax.legend(fontsize=6, loc='upper right', ncol=4)

    ax = axes1[1]
    for j in range(0, 16, 4):
        ax.plot(times, limb_output[:, j], label=f'x{16 + j}', alpha=0.8)
    ax.set_ylabel('x Limb (girdle flex)')
    ax.legend(fontsize=6, loc='upper right', ncol=4)

    ax = axes1[2]
    ax.plot(times, drive_log, 'k-', label='drive d')
    ax.set_ylabel('Drive d')
    ax.set_xlabel('Time [s]')
    ax.legend()
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(
        4, 1, figsize=(12, 10), sharex=True,
        num='Network dynamics vs time',
    )
    axes2[0].plot(times, freqs_log[:, 0], 'b-', label='Body freq')
    axes2[0].plot(times, freqs_log[:, 16], 'b--', label='Limb freq')
    axes2[0].set_ylabel('Freq [rad/s]')
    axes2[0].legend()
    axes2[0].set_title('Oscillator frequencies')

    axes2[1].plot(times, amplitudes_log[:, 0], 'g-', label='Body r')
    axes2[1].plot(times, amplitudes_log[:, 16], 'g--', label='Limb r')
    axes2[1].set_ylabel('Amplitude r')
    axes2[1].legend()

    axes2[2].plot(times, nominal_amp_log[:, 0], 'r-', label='Body R (nominal)')
    axes2[2].plot(times, nominal_amp_log[:, 16], 'r--', label='Limb R (nominal)')
    axes2[2].set_ylabel('Nominal amp R')
    axes2[2].legend()

    axes2[3].plot(times, drive_log, 'k-')
    axes2[3].set_ylabel('Drive d')
    axes2[3].set_xlabel('Time [s]')
    fig2.tight_layout()

    # Static sweep: oscillator properties vs drive (helpful sanity check)
    from robot_parameters import RobotParameters
    drives = np.linspace(0.0, 6.0, 300)
    f_body_arr = np.zeros_like(drives)
    f_limb_arr = np.zeros_like(drives)
    R_body_arr = np.zeros_like(drives)
    R_limb_arr = np.zeros_like(drives)
    for k, d in enumerate(drives):
        rp_tmp = RobotParameters(SimulationParameters(drive=float(d)))
        f_body_arr[k] = rp_tmp.freqs[0]
        f_limb_arr[k] = rp_tmp.freqs[16]
        R_body_arr[k] = rp_tmp.nominal_amplitudes[0]
        R_limb_arr[k] = rp_tmp.nominal_amplitudes[16]

    fig3, axes3 = plt.subplots(
        2, 1, figsize=(8, 6),
        num='Oscillator properties vs drive',
    )
    axes3[0].plot(drives, f_body_arr, 'b-', label='Body')
    axes3[0].plot(drives, f_limb_arr, 'b--', label='Limb')
    axes3[0].set_ylabel(r'$\omega$ [rad/s]')
    axes3[0].set_xlabel('Drive d')
    axes3[0].legend()
    axes3[0].set_title('Frequency vs drive')

    axes3[1].plot(drives, R_body_arr, 'g-', label='Body')
    axes3[1].plot(drives, R_limb_arr, 'g--', label='Limb')
    axes3[1].set_ylabel('R')
    axes3[1].set_xlabel('Drive d')
    axes3[1].legend()
    axes3[1].set_title('Nominal amplitude vs drive')
    fig3.tight_layout()

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """Exercise 1: Network & Dynamics (no MuJoCo)

    Part A: sanity-check at constant drive (short run)
    Part B: drive ramp 0 -> 6 over 20 seconds
    """

    # Part A: short constant-drive run
    run_network(duration=5, drive=SimulationParameters().drive, timestep=timestep)

    # Part B: ramp(0 to 6 over 20 seconds)
    duration_ramp = 20.0
    times = np.arange(0, duration_ramp, timestep)
    drive_profile = np.linspace(0.0, 6.0, len(times))
    run_network(
        duration=duration_ramp,
        update=True,
        drive=drive_profile,
        timestep=timestep,
    )

    # Save plots (Agg backend is non-interactive, plt.show() is not supported)
    save_figures()
    return


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())

