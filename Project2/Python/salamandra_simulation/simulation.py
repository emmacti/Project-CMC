"""Simulation"""

import os
import pickle
from multiprocessing import Pool
# import itertools
# from concurrent.futures import ProcessPoolExecutor, as_completed

import tqdm
import numpy as np

from farms_core import pylog
from farms_core.model.options import ArenaOptions
from farms_core.simulation.options import Simulator, SimulationOptions
from farms_core.experiment.options import ExperimentOptions
from farms_sim.simulation import simulation_setup

from salamandra_simulation.options import SalamandraOptions
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.controller import SalamandraController
# from salamandra_simulation.callbacks import SwimmingCallback
# from salamandra_simulation.camera import CameraCallback, save_video
from network import SalamandraNetwork


def simulation(
        sim_parameters,
        experiment_config='cmc_project_pack/configs/experiment_config.yaml',
        arena='water',
        verbose=False,
        **kwargs,
):
    """Main"""

    # Verbosity
    if not verbose:
        pylog.set_level('warning')

    # Simulation options
    output = kwargs.pop('output', os.path.join('logs', 'unassigned'))
    os.makedirs(output, exist_ok=True)
    record = kwargs.pop('record', False)
    record_path = kwargs.pop('record_path', os.path.join(output, 'video.mp4'))
    animat_id = kwargs.pop('animat_id', 0)  # None for fixed camera
    record_fps = kwargs.pop('record_fps', 30)
    record_speed = kwargs.pop('record_speed', 1.0)
    record_azimuth = kwargs.pop('record_aziomuth', -30)
    record_elevation = kwargs.pop('record_elevation', -15)
    record_distance = kwargs.pop('record_distance', 2)
    record_angular_velocity = kwargs.pop('record_angular_velocity', 0)
    record_offset = kwargs.pop('record_offset', [0, 0, 0])
    pylog.info('Setting simulation options')
    n_iterations = int(sim_parameters.duration/sim_parameters.timestep)
    exp_options: ExperimentOptions = ExperimentOptions.load(experiment_config)
    simulation_options: SimulationOptions = exp_options.simulation
    simulation_options.runtime.n_iterations = n_iterations
    simulation_options.runtime.buffer_size = n_iterations
    simulation_options.runtime.fast = kwargs.pop('fast', False)
    simulation_options.runtime.headless = kwargs.pop('headless', False)
    simulation_options.physics.timestep = sim_parameters.timestep
    simulation_options.physics.num_sub_steps = 1
    assert not kwargs, kwargs

    # Animat options
    animat_options = exp_options.animats[0]
    animat_options.spawn.pose[:3] = sim_parameters.spawn_position
    animat_options.spawn.pose[3:] = sim_parameters.spawn_orientation

    # Arena options
    water_height = None
    if arena == 'water':
        water_height = 0
        arena_path = 'cmc_project_pack/configs/arena_water_config.yaml'
        arena_options = ArenaOptions.load(arena_path)
    elif arena == 'amphibious':
        water_height = -0.1
        arena_path = 'cmc_project_pack/configs/arena_amphibious_config.yaml'
        arena_options = ArenaOptions.load(arena_path)
    elif arena == 'land':
        arena_path = 'cmc_project_pack/configs/arena_flat_config.yaml'
        arena_options = ArenaOptions.load(arena_path)
    else:
        raise ValueError(
            'arena variable should be set to: water, land or amphibious')
    if arena_options.water.sdf:
        arena_options.water.height = water_height
    else:
        swimming_extension = 'farms_mujoco.swimming.extension.SwimmingExtension'
        animat_options.extensions = [
            extension
            for extension in animat_options.extensions
            if not extension['loader'] == swimming_extension
        ]
    exp_options.arenas[0] = arena_options

    # Data
    animat_data = SalamandraData.from_options(
        animat_options=animat_options,
        simulation_options=simulation_options,
    )

    # Network
    network = SalamandraNetwork(
        sim_parameters=sim_parameters,
        n_iterations=n_iterations,
        data=animat_data,
        # state=animat_data.state,
    )

    # Other options
    options = {}

    # Simulation extensions
    for extension in simulation_options.extensions:
        match extension['loader']:
            case 'farms_core.simulation.extensions.ExperimentLogger':
                extension['config']['log_path'] = output
            case 'farms_core.simulation.extensions.ExperimentOptionsLogger':
                extension['config']['log_path'] = output
            case 'farms_mujoco.simulation.extensions.MjcfSaver':
                extension['config']['path'] = f'{output}/simulation_mjcf.xml'
    if record:
        simulation_options.extensions.append({
            "loader": "farms_mujoco.sensors.camera.CameraRecording",
            "config": {
                "path": record_path,
                "animat_id": animat_id,
                "fps": record_fps,
                "speed": record_speed,
                "azimuth": record_azimuth,
                "elevation": record_elevation,
                "distance": record_distance,
                "angular_velocity": record_angular_velocity,
                "offset": record_offset,
                "resolution": [1280, 720],
            },
        })
    if arena != 'land':
        animat_options.extensions.append({
            "loader": "farms_mujoco.swimming.extension.SwimmingExtension",
            "config": {"water_properties": None},
        })

    # Setup simulation
    pylog.info('Setting up simulation')
    sim = simulation_setup(experiment_options=exp_options, **options)

    # Set up controller
    pylog.info('Setting up controller')
    for extension in sim.task.extensions:
        if isinstance(extension, SalamandraController):
            pylog.info('Replacing controller')
            extension.network = network

    # Simulation parameters
    with open(
            file=os.path.join(output, 'sim_parameters.pickle'),
            mode='wb',
    ) as param_file:
        pickle.dump(sim_parameters, param_file)

    # Run simulation
    pylog.info('Running simulation')
    sim.run()

    if not verbose:
        pylog.set_level('debug')

    return sim, animat_data


def simulation_parallel_interface(parameters):
    """Simulation with single argument"""
    simulation(**parameters)


def simulation_sweep(arguments, processes=8):
    """Simulation sweep

    Number of processes can be updated according to computer resources available

    """
    with Pool(processes=processes) as pool:
        pool.map(simulation_parallel_interface, arguments)

