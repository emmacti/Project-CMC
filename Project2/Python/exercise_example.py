"""Exercise example"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt

from farms_core.utils.profile import profile
from farms_amphibious.data.data import AmphibiousExperimentData


def load_data(
        log_files: str,
        simulation_i: int,
) -> tuple[AmphibiousExperimentData, SimulationParameters]:
    """Load data"""
    experiment_data_file = os.path.join(
        log_files.format(simulation_i),
        'simulation.hdf5',
    )
    exp_data = AmphibiousExperimentData.from_file(experiment_data_file)
    sim_parameters_file = os.path.join(
        log_files.format(simulation_i),
        'sim_parameters.pickle',
    )
    with open(sim_parameters_file, 'rb') as param_file:
        parameters = pickle.load(param_file)
    return exp_data, parameters


def exercise_example(timestep, n_simulations=1):
    """Exercise example"""

    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            # Orientation in Euler angles [rad]
            spawn_orientation=[0, 0, np.pi/2],
            drive=drive,  # An example of parameter part of the grid search
            amplitudes=[1, 2, 3],  # Just an example
            phase_lag_body=0,  # or np.zeros(n_joints) for example
            turn=0,  # Another example
            # ...
        )
        for drive in np.linspace(3, 4, n_simulations)
        # for amplitudes in ...
        # for ...
    ]

    # Run simulations
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'water', give it a try!
            # fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
            output=f'logs/example/sim_{simulation_i}',
            record=True,  # Record video
            # video savging path
            record_path=f"logs/example/video_{simulation_i}.mp4",
            verbose=True,
        )


def example_load_data():
    """Example to show how to load data"""
    exp_data, parameters = load_data(
        log_files="logs/example/sim_{}",
        simulation_i=0,
    )
    data = exp_data.animats[0]
    timestep = exp_data.timestep
    n_iterations = np.shape(data.sensors.links.array)[0]
    times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
    )
    drive = parameters.drive  # Example to get parameters
    phase_lag_body = parameters.phase_lag_body
    joints_pos = np.array(data.sensors.joints.positions_all())
    for name, data in zip(data.sensors.joints.names, joints_pos.T):
        plt.plot(times, data, label=name)
    plt.xlabel('Time [s]')
    plt.ylabel('Joints position [rad]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Open Ipython to interact with the code (uv pip install ipython)
    # This can be useful for exploring the contents of data.sensors for example
    # from IPython import embed; embed()


if __name__ == '__main__':
    # exercise_example(timestep=5e-3)
    profile(exercise_example, timestep=5e-3)
    example_load_data()

