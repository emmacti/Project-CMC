"""[Project2] Exercise 1: Implement & run network without MuJoCo"""

import time
from dataclasses import dataclass

import numpy as np
from pathlib import Path
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


def run_network(duration, update=False, drive=0, timestep=1e-2, output_folder="logs/exercise_p1", label="run"):
    """Run network without MuJoCo and plot results

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

    # --- Exercise 1B: make drive a vector that ramps from 0 to 6 ---
    if np.isscalar(drive):
        drive_vector = np.full(n_iterations, float(drive))
    else:
        drive_vector = np.asarray(drive, dtype=float)
        assert len(drive_vector) == n_iterations, (
            "drive array must have length n_iterations")

    sim_parameters = SimulationParameters(
        drive=drive_vector[0],      # initialise with first value
        amplitude_gradient=None,
        phase_lag_body=None,
    )

    state = SalamandraState.salamandra_robot(n_iterations, n_oscillators=32)
    network = SalamandraNetwork(
        sim_parameters,
        n_iterations,
        DataState(state=state),
    )

    osc_left  = np.arange(0, 16, 2)   # body left oscillators
    osc_right = np.arange(1, 16, 2)   # body right oscillators
    osc_legs  = np.arange(16, 32)     # limb oscillators

    # Logs
    phases_log = np.zeros([n_iterations, len(network.state.phases(iteration=0))])
    phases_log[0, :] = network.state.phases(iteration=0)

    amplitudes_log = np.zeros([n_iterations, len(network.state.amplitudes(iteration=0))])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)

    freqs_log = np.zeros([n_iterations, len(network.robot_parameters.freqs)])
    freqs_log[0, :] = network.robot_parameters.freqs

    drive_log = np.zeros(n_iterations)
    drive_log[0] = drive_vector[0]

    nominal_amp_log = np.zeros([n_iterations, network.robot_parameters.n_oscillators])
    nominal_amp_log[0, :] = network.robot_parameters.nominal_amplitudes

    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        # Update drive at each step (Exercise 1B: ramping drive)
        current_drive = drive_vector[i + 1]

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
        n_iterations, toc - tic))

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------

    # --- Plot 1: Body oscillator motor outputs ---
    body_output = amplitudes_log[:, :16] * (1 + np.cos(phases_log[:, :16]))
    limb_output = amplitudes_log[:, 16:] * (1 + np.cos(phases_log[:, 16:]))

    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                num=f'{label}_motor_outputs')
    fig1.suptitle('Body oscillator motor outputs', fontsize=12)

    ax = axes1[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for k, j in enumerate(range(0, 16, 2)):   # left body oscillators (even)
        ax.plot(times, body_output[:, j], color=colors[k],
                label=f'x{j}', alpha=0.85)
    ax.set_ylabel('x Body (left)')
    ax.legend(fontsize=7, loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3)

    ax = axes1[1]
    limb_labels = ['FL girdle', 'FR girdle', 'HL girdle', 'HR girdle']
    for k, j in enumerate(range(0, 16, 4)):   # one girdle flexor per limb
        ax.plot(times, limb_output[:, j], label=f'x{16+j} ({limb_labels[k]})',
                alpha=0.85)
    ax.set_ylabel('x Limb (girdle flex)')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes1[2]
    ax.plot(times, drive_log, 'k-', linewidth=1.5, label='drive d')
    ax.set_ylabel('Drive d')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Plot 2: Frequencies and amplitudes vs time ---
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                                num=f'{label}_network_dynamics')

    axes2[0].plot(times, freqs_log[:, 0], 'b-', label='Body freq')
    axes2[0].plot(times, freqs_log[:, 16], 'b--', label='Limb freq')
    axes2[0].set_ylabel('Freq [Hz]')
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

    # --- Plot 3: Frequency vs drive (static sweep) ---
    drives = np.linspace(0, 6, 300)
    f_body_arr = np.zeros_like(drives)
    f_limb_arr = np.zeros_like(drives)
    R_body_arr = np.zeros_like(drives)
    R_limb_arr = np.zeros_like(drives)

    for k, d in enumerate(drives):
        p = SimulationParameters(drive=d)
        from robot_parameters import RobotParameters
        rp_tmp = RobotParameters(p)
        f_body_arr[k] = rp_tmp.freqs[0]
        f_limb_arr[k] = rp_tmp.freqs[16]
        R_body_arr[k] = rp_tmp.nominal_amplitudes[0]
        R_limb_arr[k] = rp_tmp.nominal_amplitudes[16]

    fig3, axes3 = plt.subplots(2, 1, figsize=(8, 6),
                                num=f'{label}_properties_vs_drive')
    axes3[0].plot(drives, f_body_arr, 'b-', label='Body')
    axes3[0].plot(drives, f_limb_arr, 'b--', label='Limb')
    axes3[0].set_ylabel('v [Hz]')
    axes3[0].set_xlabel('Drive d')
    axes3[0].legend()
    axes3[0].set_title('Frequency vs drive (cf. Fig 5A)')
    axes3[0].axvline(1.0, color='gray', linestyle=':')
    axes3[0].axvline(3.0, color='gray', linestyle=':')
    axes3[0].axvline(5.0, color='gray', linestyle=':')

    axes3[1].plot(drives, R_body_arr, 'g-', label='Body')
    axes3[1].plot(drives, R_limb_arr, 'g--', label='Limb')
    axes3[1].set_ylabel('R')
    axes3[1].set_xlabel('Drive d')
    axes3[1].legend()
    axes3[1].set_title('Nominal amplitude vs drive (cf. Fig 5B)')
    axes3[1].axvline(1.0, color='gray', linestyle=':')
    axes3[1].axvline(3.0, color='gray', linestyle=':')
    axes3[1].axvline(5.0, color='gray', linestyle=':')
    fig3.tight_layout()

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 2] Exercise 1A: Full CPG network (axial + limbs)"""

    # Exercise 1A: static drive = 2.0 (walking regime)
    pylog.info('Running Exercise 1A: walking drive d=2.0')
    run_network(duration=10, drive=2.0, timestep=timestep, label="ex1a_drive_2")

    # Exercise 1B: ramping drive from 0 to 6 over 20 s
    pylog.info('Running Exercise 1B: ramping drive 0→6 over 20 s')
    duration_ramp = 20.0
    times_ramp = np.arange(0, duration_ramp, timestep)
    drive_ramp = np.linspace(0, 6, len(times_ramp))
    run_network(duration=duration_ramp, drive=drive_ramp, timestep=timestep, label="ex1b_ramp_0_to_6")

    save_figures(extensions=['png'])
    pylog.info("Plots saved in results/")


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())
