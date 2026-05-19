"""[Project2] Exercise 2: Walking & Gait Transitions with Salamander Robot"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from salamandra_simulation.simulation import simulation
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from farms_amphibious.data.data import AmphibiousExperimentData


# Helpers

def load_data(log_path):
    """Load experiment data and sim_parameters from a log directory."""
    exp_data = AmphibiousExperimentData.from_file(
        os.path.join(log_path, 'simulation.hdf5')
    )
    with open(os.path.join(log_path, 'sim_parameters.pickle'), 'rb') as f:
        sim_params = pickle.load(f)
    return exp_data, sim_params


def plot_joint_positions(log_path, fig_title, timestep):
    """Plot joint positions of body and limb joints over time."""
    exp_data, sim_params = load_data(log_path)
    data = exp_data.animats[0]
    joints_pos = np.array(data.sensors.joints.positions_all())
    n_iterations = joints_pos.shape[0]
    times = np.arange(n_iterations) * timestep

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, num=fig_title)

    # Body joints (first 8)
    ax = axes[0]
    for j in range(min(8, joints_pos.shape[1])):
        name = data.sensors.joints.names[j] if hasattr(data.sensors.joints, 'names') else f'joint_{j}'
        ax.plot(times, joints_pos[:, j], label=str(name), alpha=0.8)
    ax.set_ylabel('Body joint position [rad]')
    ax.legend(fontsize=6, loc='upper right', ncol=4)
    ax.set_title(fig_title)
    ax.grid(True)

    # Limb joints (remaining)
    ax = axes[1]
    for j in range(8, min(16, joints_pos.shape[1])):
        name = data.sensors.joints.names[j] if hasattr(data.sensors.joints, 'names') else f'joint_{j}'
        ax.plot(times, joints_pos[:, j], label=str(name), alpha=0.8)
    ax.set_ylabel('Limb joint position [rad]')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=6, loc='upper right', ncol=4)
    ax.grid(True)

    fig.tight_layout()


def compute_speed(log_path, timestep):
    """Compute mean forward speed from GPS link positions."""
    exp_data, _ = load_data(log_path)
    data = exp_data.animats[0]
    # Link 0 = head; use x-coordinate (forward direction)
    gps = np.array(data.sensors.links.urdf_positions())  # [n_iter, n_links, 3]
    head_x = gps[:, 0, 0]
    total_dist = head_x[-1] - head_x[0]
    duration = len(head_x) * timestep
    speed = total_dist / duration
    return speed


# Exercise 2A: Walking with default (fixed) drive

def exercise_walk(timestep):
    """Exercise 2A: Walking with constant drive in the walking regime (d=2.0).

    Deliverable: 10-15 s video showing the robot walking with default parameters.
    Arena: land (flat ground).
    """
    log_dir = 'logs/2a_walk'
    os.makedirs(log_dir, exist_ok=True)

    # Walking drive: d=2.0 is well inside the walking regime (1 < d < 3)
    drive = 2.0
    duration = 15  # 15 s → satisfies the 10-15 s video requirement

    sim_parameters = SimulationParameters(
        duration=duration,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi / 2],
        drive=drive,
        phase_lag_body= 2 * np.pi / 8,   # default travelling wave
        amplitude_gradient=None,
    )

    sim, animat_data = simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=False,
        headless=False,          # set True for batch runs
        output=log_dir,
        record=True,
        record_path=os.path.join(log_dir, '2a_walking_default.mp4'),
        verbose=False,
    )

    # Plot joint trajectories
    plot_joint_positions(log_dir, '2A: Walking (d=2.0)', timestep)
    return


# Drive ramp in WATER (swimming)

def exercise_ramp_swim(timestep):
    """Exercise 2B: Drive ramp experiment in WATER.

    Drive linearly increases from 0 to 6 over ~40 s.
    Sequence expected:
      0–1   : sub-threshold → no oscillation (body still)
      1–3   : walking regime → active limbs + body standing wave
      3–5   : swimming regime → body travelling wave, limbs silent
      5–6   : above saturation → saturated swimming
    Deliverable: ~40 s video (side view).
    """
    log_dir = 'logs/2b_ramp_swim'
    os.makedirs(log_dir, exist_ok=True)

    duration = 40  # seconds

    # drive_array is passed as a scalar; we update it each step via step()
    # in robot_parameters.  Here we encode the ramp via drive_ramp_end and
    # drive_ramp_start in SimulationParameters, which robot_parameters.step()
    # can use to linearly interpolate the drive.
    sim_parameters = SimulationParameters(
        duration=duration,
        timestep=timestep,
        spawn_position=[0, 0, 0.0],    # start at water level
        spawn_orientation=[0, 0, np.pi / 2],
        drive=0.0,                      # initial drive (will ramp)
        drive_ramp_start=0.0,
        drive_ramp_end=6.0,
        drive_ramp_duration=duration,   # ramp over full simulation
        phase_lag_body=2 * np.pi / 8,
        amplitude_gradient=None,
    )

    sim, animat_data = simulation(
        sim_parameters=sim_parameters,
        arena='water',
        fast=False,
        headless=False,
        output=log_dir,
        record=True,
        record_path=os.path.join(log_dir, '2b_ramp_swim.mp4'),
        #record_azimuth=-90,             # side view
        #record_elevation=0,
        verbose=False,
    )

    # Plot joint trajectories
    plot_joint_positions(log_dir, '2B: Ramp drive in water', timestep)
    return


# Drive ramp on LAND (walking → swimming transition)

def exercise_ramp_walk(timestep):
    """Exercise 2C: Drive ramp experiment on LAND (flat ground).

    Same ramp as 2B but on flat ground.
    Sequence expected:
      0–1   : sub-threshold → body still, limbs inactive
      1–3   : walking gait → trot with body undulation
      3–5   : high drive → body switches to travelling wave, limbs go silent
               (swimming pattern on land — ineffective but visible)
    Deliverable: ~40 s video (side view).
    """
    log_dir = 'logs/2c_ramp_walk'
    os.makedirs(log_dir, exist_ok=True)

    duration = 40

    sim_parameters = SimulationParameters(
        duration=duration,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi / 2],
        drive=0.0,
        drive_ramp_start=0.0,
        drive_ramp_end=6.0,
        drive_ramp_duration=duration,
        phase_lag_body=2 * np.pi / 8,
        amplitude_gradient=None,
    )

    sim, animat_data = simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=False,
        headless=False,
        output=log_dir,
        record=True,
        record_path=os.path.join(log_dir, '2c_ramp_walk.mp4'),
        #record_azimuth=-90,             # side view
        #record_elevation=0,
        verbose=False,
    )

    # Plot joint trajectories
    plot_joint_positions(log_dir, '2C: Ramp drive on land', timestep)
    return

# Entry point

if __name__ == '__main__':
    timestep = 5e-3
    exercise_walk(timestep=timestep)
    exercise_ramp_swim(timestep=timestep)
    exercise_ramp_walk(timestep=timestep)

    if not save_plots():
        plt.show()
    else:
        save_figures()

