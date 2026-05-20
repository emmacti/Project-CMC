"""Exercise 4: Transitions between swimming and walking (Project 2 Part B)."""

import os

import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog


def _transition_params(timestep, duration=40.0, **kwargs):
    """Parameters for amphibious gait-transition simulations."""
    base = dict(
        duration=duration,
        timestep=timestep,
        drive=2.0,
        phase_lag_body=2 * np.pi / 8,
        update_drive=True,
        walk_drive=2.0,
        swim_drive=4.5,
        feet_contact_on=0.12,
        feet_contact_off=0.04,
    )
    base.update(kwargs)
    return SimulationParameters(**base)


def exercise_4a_transition(timestep):
    """Exercise 4: land↔water transitions on the amphibious ramp.

    Feedback strategy (implemented in robot_parameters.step):
    - Foot contact sensors estimate ground support.
    - Hysteresis: feet above threshold → walking drive (d≈2);
      feet below threshold → swimming drive (d≈4.5).
    - Biologically inspired: limb mechanoreception / ground reaction
      cues gate the MLR drive level (Ijspeert 2007, Crespi 2013).
    """
    # Land → water: start on dry ramp, walk toward pool
    log_walk2swim = 'logs/ex4_walk2swim'
    os.makedirs(log_walk2swim, exist_ok=True)
    sim_walk2swim = _transition_params(
        timestep,
        spawn_position=[4.0, 0.0, 0.12],
        spawn_orientation=[0, 0, np.pi],
        drive=2.0,
    )
    pylog.info('Exercise 4: walk → swim transition')
    simulation(
        sim_parameters=sim_walk2swim,
        arena='amphibious',
        fast=False,
        headless=False,
        output=log_walk2swim,
        record=True,
        record_path=os.path.join(log_walk2swim, 'ex4_walk2swim.mp4'),
        verbose=False,
    )

    # Water → land: start in water, swim toward ramp
    log_swim2walk = 'logs/ex4_swim2walk'
    os.makedirs(log_swim2walk, exist_ok=True)
    sim_swim2walk = _transition_params(
        timestep,
        spawn_position=[-3.5, 0.0, -0.05],
        spawn_orientation=[0, 0, 0.0],
        drive=4.5,
    )
    pylog.info('Exercise 4: swim → walk transition')
    simulation(
        sim_parameters=sim_swim2walk,
        arena='amphibious',
        fast=False,
        headless=False,
        output=log_swim2walk,
        record=True,
        record_path=os.path.join(log_swim2walk, 'ex4_swim2walk.mp4'),
        verbose=False,
    )


if __name__ == '__main__':
    exercise_4a_transition(timestep=5e-3)
