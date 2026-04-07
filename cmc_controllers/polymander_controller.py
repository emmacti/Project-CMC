"""Polymander controller"""

from typing import List
import numpy as np

from dm_control.rl.control import Task
from dm_control.mjcf.physics import Physics

from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.experiment.options import ExperimentOptions

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.control.amphibious import JointMuscleController
from farms_amphibious.control.network import AnimatNetwork


class NeuralNetwork(AnimatNetwork):
    """Neural Network"""

    def __init__(
        self,
        data: AnimatData,
        **kwargs,
    ):
        super().__init__(
            data=data,
            n_iterations=np.shape(data.state.array)[0],
        )
        self.initialize_episode()

    def initialize_episode(self):
        """Initialize episode"""
        self.data.state.array[1:, :] = 0

    def step(
        self,
        iteration: int,
        time: float,
        timestep: float,
        checks: bool = False,
        strict: bool = False,
    ):
        """Control step

        Called after obtaining all the current sensor data, and right before
        calling the physics.

        """
        raise NotImplementedError


class NeuralController(JointMuscleController):
    """Neural controller base class"""

    def __init__(
            self,
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
            animat_network: AnimatNetwork,
            n_iterations=0,
    ):
        super().__init__(
            animat_options=animat_options,
            animat_data=animat_data,
            animat_network=animat_network,
            animat_i=0,
            substep=True,
        )
        self.n_iterations = n_iterations

    @classmethod
    def from_options(
        cls,
        config: dict,
        experiment_options: ExperimentOptions,
        animat_i: int,
        animat_data: AnimatData,
        animat_options: AnimatOptions,
    ):
        """From options

        animat_options = experiment_options.animats[animat_i]

        """
        del animat_i
        animat_network = NeuralNetwork(
            data=animat_data,
        )

        return cls(
            animat_options=animat_options,
            animat_data=animat_data,
            animat_network=animat_network,
            n_iterations=experiment_options.simulation.runtime.n_iterations,
        )

    def initialize_episode(self, task: Task, physics: Physics):
        """Initialize episode"""
        self.animat_data.sensors.links.array[1:, :, :] = 0
        self.animat_data.sensors.joints.array[1:, :, :] = 0
        self.animat_data.sensors.contacts.array[1:, :, :] = 0
        self.animat_data.sensors.xfrc.array[1:, :, :] = 0
        self.network.initialize_episode()

    def before_step(self, task: Task, action, physics: Physics):
        """Before step"""
        del action
        time = physics.time()
        # timestep = physics.timestep()
        timestep = task.timestep
        index = task.iteration % task.buffer_size
        self.step(iteration=index, time=time, timestep=timestep)

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ):
        self.network.step(iteration, time, timestep)
        for net2joints in self.network2joints.values():
            net2joints.step(iteration)


class PolymanderController(NeuralController):
    """Polymander Controller Base Class"""

    def __init__(self,
                 animat_options: AmphibiousOptions,
                 animat_data: AmphibiousData,
                 animat_network: AnimatNetwork,):
        # super().__init__(animat_data, animat_options, experiment_options, n_joints, n_iterations)
        super().__init__(animat_options,
                         animat_data,
                         animat_network,)

