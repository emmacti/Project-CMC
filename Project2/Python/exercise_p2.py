"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — avoids GUI errors on macOS
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_walk(timestep):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    os.makedirs('./logs/ex2_walk/', exist_ok=True)
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.08],
        spawn_orientation=[0, 0, np.pi/2],
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=False,
        record=True,
        output='logs/ex2_walk/sim_0',
        record_path='logs/ex2_walk/walk.mp4',
        record_aziomuth=0,    # side view
        record_elevation=-5,
        record_distance=2.5,
        verbose=True,
    )
    return


def exercise_ramp_swim(timestep):
    "[Project 1] Q2 Swimming with an increasing (ramp) drive"
    os.makedirs('./logs/ex2_ramp_swim/', exist_ok=True)
    duration = 40
    n_iterations = int(duration/timestep)
    drive = np.linspace(0.0, 6.0, n_iterations)
    sim_parameters = SimulationParameters(
        duration=duration,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=drive,
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='water',
        fast=False,
        record=True,
        output='logs/ex2_ramp_swim/sim_0',
        record_path='logs/ex2_ramp_swim/ramp_swim.mp4',
        record_aziomuth=0,
        record_elevation=-5,
        record_distance=2.5,
        verbose=False,
    )
    return


def exercise_ramp_walk(timestep):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    os.makedirs('./logs/ex2_ramp_walk/', exist_ok=True)
    duration = 40
    n_iterations = int(duration/timestep)
    drive = np.linspace(0.0, 6.0, n_iterations)
    sim_parameters = SimulationParameters(
        duration=duration,
        timestep=timestep,
        spawn_position=[0, 0, 0.08],
        spawn_orientation=[0, 0, np.pi/2],
        drive=drive,
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=False,
        record=True,
        output='logs/ex2_ramp_walk/sim_0',
        record_path='logs/ex2_ramp_walk/ramp_walk.mp4',
        record_aziomuth=0,    # side view: perpendicular to Y-axis motion
        record_elevation=-5,
        record_distance=2.5,
        verbose=False,
    )
    return


if __name__ == '__main__':
    exercise_walk(timestep=5e-3)
    exercise_ramp_swim(timestep=5e-3)
    exercise_ramp_walk(timestep=5e-3)

