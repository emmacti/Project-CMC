"""Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from farms_core import pylog


def exercise_4a_transition(timestep):
    """4a Transitions

    In this exerices, we will implement transitions.
    The salamander robot needs to perform swimming to walking
    and walking to swimming transitions.

    Hint:
        - The handling of the drive update is done in robot_parameters.py
        - Set the  arena to 'amphibious'
        - Use the contacts values to find the point where
          the robot should transition
        - Simulation can be stopped/played in the middle
          by pressing the space bar
        - Printing or debug mode of vscode can be used
          to understand the sensor values

    We recommend using the following in robot_parameters.py::step():

    index = 0 if iteration == 0 else (iteration - 1)
    contacts_all = np.linalg.norm(np.array(
        salamandra_data.sensors.contacts.totals()[index]
    ), axis=1)
    contacts_body = contacts_all[:9]
    contacts_upper_limbs = contacts_all[9:17:2]
    contacts_feet = contacts_all[10:18:2]

    # Use self.update_drive = parameters.update_drive in __init__
    if self.update_drive:
        ...

    """
    # Use exercise_example.py for reference
    # Additional hints:
    # sim_parameters = SimulationParameters(
    #     ...,
    #     spawn_position=[4, 0, 0.0],
    #     spawn_orientation=[0, 0, np.pi],
    # )
    # _sim, _data = simulation(
    #     sim_parameters=sim_parameters,
    #     arena='amphibious',
    #     fast=True,
    #     record=True,
    #     record_path='walk2swim',  # or swim2walk
    # )
    os.makedirs('./logs/ex4_transition/', exist_ok=True)

    # Walk -> swim: start on land, pointed towards water
    sim_parameters = SimulationParameters(
        duration=35,
        timestep=timestep,
        drive=2.5,
        update_drive=True,
        drive_walk=2.5,
        drive_swim=4.5,
        transition_contact_threshold=0.6,
        transition_hysteresis=0.2,
        spawn_position=[2.0, 0.0, 0.05],
        spawn_orientation=[0, 0, np.pi],  
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='amphibious',
        fast=False,
        record=True,
        output='logs/ex4_transition/walk2swim_sim_0',
        record_path='logs/ex4_transition/walk2swim.mp4',
        verbose=False,
    )

    # Swim -> walk: start in water, pointed towards land
    sim_parameters = SimulationParameters(
        duration=35,
        timestep=timestep,
        drive=4.5,
        update_drive=True,
        drive_walk=2.5,
        drive_swim=4.5,
        transition_contact_threshold=0.6,
        transition_hysteresis=0.2,
        spawn_position=[-2.0, 0.0, 0.05],
        spawn_orientation=[0, 0, np.pi],  # flip heading 
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='amphibious',
        fast=False,
        record=True,
        output='logs/ex4_transition/swim2walk_sim_0',
        record_path='logs/ex4_transition/swim2walk.mp4',
        verbose=False,
    )
    return


if __name__ == '__main__':
    exercise_4a_transition(timestep=5e-3)

