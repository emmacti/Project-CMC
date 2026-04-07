import numpy as np
from typing import List

from farms_core.model.data import AnimatData
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.options import AnimatOptions

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.control.network import AnimatNetwork

from farms_core import pylog

from cmc_controllers.polymander_controller import PolymanderController, NeuralNetwork


class WaveNetwork(NeuralNetwork):
    """Dummy Network"""

    def __init__(
        self,
        data: AnimatData,
        freq: float,
        amp: float,
        twl: float,
        n_body_joints: int,
        left_body_idx: slice,
        right_body_idx: slice,
        **kwargs,
    ):
        super().__init__(data, **kwargs)

        # Controller state
        self.state = np.zeros((self.n_iterations, 2*n_body_joints))

        # Wave controller paramters
        self.freq = freq
        self.amp = amp
        self.twl = twl
        self.n_body_joints = n_body_joints
        self.left_body_idx = left_body_idx
        self.right_body_idx = right_body_idx

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
        # Enforce M_sum = 2 and M_diff = A*sin(...)
        # => M_L = (M_sum + M_diff)/2 = 1 + 0.5*M_diff
        # => M_R = (M_sum - M_diff)/2 = 1 - 0.5*M_diff
        i = np.arange(self.n_body_joints, dtype=float)
        phase = 2.0 * np.pi * (self.freq * time - self.twl * (i / self.n_body_joints))        m_diff = self.amp * np.sin(phase)
        act_left = 1.0 + 0.5 * m_diff
        act_right = 1.0 - 0.5 * m_diff

        # implicit muscle activation
        self.data.state.array[iteration, self.left_body_idx] = act_left
        self.data.state.array[iteration, self.right_body_idx] = act_right

        # state logging
        self.state[iteration, self.left_body_idx] = act_left
        self.state[iteration, self.right_body_idx] = act_right


class WaveController(PolymanderController):
    """WaveController"""

    def __init__(self,
                 animat_options: AmphibiousOptions,
                 animat_data: AmphibiousData,
                 config):

        control_joint_names = [
            joint.joint_name for joint in animat_options.control.motors]
        body_joint_names = [
            name for name in control_joint_names if "body" in name and 'passive' not in name]
        leg_joint_names = [
            name for name in control_joint_names if "leg" in name]

        self.n_body_joints = len(body_joint_names)
        self.n_leg_joints = len(leg_joint_names)

        self.left_body_idx = slice(0, 2*self.n_body_joints, 2)
        self.right_body_idx = slice(1, 2*self.n_body_joints+1, 2)
        self.left_leg_idx = slice(
            2 *
            self.n_body_joints,
            2 *
            self.n_body_joints +
            2 *
            self.n_leg_joints,
            2)
        self.right_leg_idx = slice(
            2 *
            self.n_body_joints +
            1,
            2 *
            self.n_body_joints +
            2 *
            self.n_leg_joints +
            1,
            2)

        animat_network = WaveNetwork(data=animat_data,
                                     freq=config['freq'],
                                     amp=config['amp'],
                                     twl=config['twl'],
                                     n_body_joints=self.n_body_joints,
                                     left_body_idx=self.left_body_idx,
                                     right_body_idx=self.right_body_idx,
                                     )

        super().__init__(
            animat_options=animat_options,
            animat_data=animat_data,
            animat_network=animat_network,
        )

        self.config = config

    @classmethod
    def from_options(
        cls,
        config: dict,
        experiment_options: ExperimentOptions,
        animat_i: int,
        animat_data: AnimatData,
        animat_options: AnimatOptions,
    ):
        del animat_i
        return cls(
            animat_options=animat_options,
            animat_data=animat_data,
            config=config,
        )

