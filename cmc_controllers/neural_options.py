"""Animat options"""

from typing import List, Dict, Union
from enum import Enum
from functools import partial
from itertools import product

import numpy as np

from farms_core.options import Options
from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.options import SimulationOptions
from farms_core.model.options import (
    AnimatOptions,
    SpawnOptions,
    ControlOptions,
    ArenaOptions,
    AnimatExtensionOptions,
)
from farms_amphibious.model.convention import AmphibiousConvention
from farms_amphibious.model.options import (
    KinematicsControlOptions,
    AmphibiousMorphologyOptions,
    AmphibiousMotorOptions,
    AmphibiousMotorTransformOptions,
    AmphibiousMotorOffsetOptions,
    AmphibiousSensorsOptions,
    AmphibiousMuscleSetOptions,
    AmphibiousAdhesionsOptions,
    AmphibiousVisualsOptions,
    AmphibiousPassiveJointOptions,

)

# pylint: disable=too-many-lines,too-many-arguments,
# pylint: disable=too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-instance-attributes


def options_kwargs_float_keys():
    """Options kwargs float keys"""
    return [
        'kinematics_sampling', 'kinematics_start', 'kinematics_end',
        'muscle_alpha', 'muscle_beta', 'muscle_gamma', 'muscle_delta',
    ]


def options_kwargs_float_list_keys():
    """Options kwargs float list keys"""
    return ['solref']


def options_kwargs_int_keys():
    """Options kwargs int keys"""
    return ['kinematics_time_index']


def options_kwargs_int_list_keys():
    """Options kwargs int list keys"""
    return ['kinematics_indices']


def options_kwargs_str_keys():
    """Options kwargs string keys"""
    return [
        'kinematics_file',
    ]


def options_kwargs_str_list_keys():
    """Options kwargs str list keys"""
    return ['collisions_list']


def options_kwargs_bool_keys():
    """Options kwargs bool keys"""
    return ['inanimate', 'kinematics_invert', 'kinematics_degrees']


def options_kwargs_animat_keys():
    """Options kwargs animat keys"""
    return (
        options_kwargs_float_keys()
        + options_kwargs_float_list_keys()
        + options_kwargs_int_keys()
        + options_kwargs_int_list_keys()
        + options_kwargs_str_keys()
        + options_kwargs_str_list_keys()
        + options_kwargs_bool_keys()
    )


def options_kwargs_sph_float_keys():
    """Options kwargs SPH float keys"""
    return [
        'sph_log_freq',
        'sph_spacing', 'sph_hdx',
        'sph_density_solid', 'sph_rho_fluid', 'sph_depth',
        'sph_multiplier_h', 'sph_multiplier_mass',
        'sph_multiplier_volume', 'sph_multiplier_rad_s',
        'sph_factor_solid', 'sph_co', 'sph_xsph_eps',
        'sph_alpha', 'sph_beta', 'sph_gamma',
    ]


def options_kwargs_sph_keys():
    """Options kwargs SPH keys"""
    return options_kwargs_sph_float_keys()


def options_kwargs_arena_keys():
    """Options kwargs arena keys"""
    return options_kwargs_sph_keys()


def options_kwargs_all_keys():
    """Options kwargs all keys"""
    return (
        options_kwargs_animat_keys()
        + options_kwargs_arena_keys()
    )


class NeuralOptions(AnimatOptions):
    """Simulation options"""

    def __init__(self, sdf: str, **kwargs):
        super().__init__(
            sdf=sdf,
            spawn=SpawnOptions(**kwargs.pop('spawn')),
            morphology=AmphibiousMorphologyOptions(**kwargs.pop('morphology')),
            control=NeuralControlOptions(**kwargs.pop('control')),
            extensions=[
                AnimatExtensionOptions(**extension)
                for extension in kwargs.pop('extensions')
            ],
        )
        self.show_xfrc = kwargs.pop('show_xfrc')
        self.scale_xfrc = kwargs.pop('scale_xfrc')
        self.mujoco = kwargs.pop('mujoco')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def default(cls):
        """Deafault options"""
        return cls.from_options({})

    @classmethod
    def from_options(cls, kwargs=None):
        """From options"""
        options = {}
        options['sdf'] = kwargs.pop('sdf_path')
        options['name'] = kwargs.pop('name', 'Animat')
        options['morphology'] = kwargs.pop(
            'morphology',
            AmphibiousMorphologyOptions.from_options(kwargs),
        )
        convention = AmphibiousConvention.from_morphology(
            morphology=options['morphology'],
            **{
                key: kwargs.get(key, False)
                for key in ('single_osc_body', 'single_osc_legs')
                if key in kwargs
            },
        )
        options['spawn'] = kwargs.pop(
            'spawn',
            SpawnOptions.from_options(kwargs)
        )
        options['mujoco'] = kwargs.pop('mujoco', {})
        if 'solref' in kwargs:
            options['mujoco']['solref'] = kwargs.pop('solref')
        options['control'] = NeuralControlOptions.from_options(kwargs)
        options['control'].defaults_from_convention(convention, kwargs)
        options['show_xfrc'] = kwargs.pop('show_xfrc', False)
        options['scale_xfrc'] = kwargs.pop('scale_xfrc', 1)
        assert not kwargs, f'Unknown kwargs: {kwargs}'
        return cls(**options)

    def state_init(self):
        """Initial states"""
        return [
            # JOINT INPUTS
            0.0 for osc in self.control.network.oscillators
        ] + [
            # JOINT OFFSETS
            joint.initial[0] for joint in self.morphology.joints
        ]


class NeuralExperimentOptions(ExperimentOptions):
    """Neural experiment options"""

    def __init__(
            self,
            simulation: str | SimulationOptions,
            animats: list[str] | list[NeuralOptions],
            arenas: list[str] | list[ArenaOptions],
    ):
        super().__init__(
            simulation=simulation,
            animats=animats,
            arenas=arenas,
        )


class NeuralControlOptions(ControlOptions):
    """Neural control options"""

    def __init__(self, **kwargs):
        super().__init__(
            controller_loader=kwargs.pop('controller_loader'),
            sensors=AmphibiousSensorsOptions(**kwargs.pop('sensors')),
            motors=[
                AmphibiousMotorOptions(**motor)
                for motor in kwargs.pop('motors')
            ],
        )
        network_options = kwargs.pop('network', None)
        self.network = (
            NeuralNetworkOptions(**network_options)
            if network_options is not None
            and 'oscillators' in network_options
            else None
        )
        self.muscles = [
            AmphibiousMuscleSetOptions(**muscle)
            for muscle in kwargs.pop('muscles')
        ]
        self.hill_muscles = kwargs.pop('hill_muscles', [])
        self.adhesions = [
            AmphibiousAdhesionsOptions(**adhesion)
            for adhesion in kwargs.pop('adhesions')
        ]
        self.visuals = [
            AmphibiousVisualsOptions(**visual)
            for visual in kwargs.pop('visuals')
        ]
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def options_from_kwargs(cls, kwargs):
        """Options from kwargs"""
        options = super(cls, cls).options_from_kwargs({
            'sensors': kwargs.pop(
                'sensors',
                AmphibiousSensorsOptions.options_from_kwargs(kwargs),
            ),
            'motors': kwargs.pop('motors', {}),
        })
        options['network'] = kwargs.pop(
            'network',
            NeuralNetworkOptions.from_options(kwargs).to_dict()
        )
        options['muscles'] = kwargs.pop('muscles', [])
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""
        self.sensors.defaults_from_convention(convention, kwargs)
        self.network.defaults_from_convention(convention, kwargs)

        # Joints
        n_joints = convention.n_joints()
        offsets = [None]*n_joints

        # Motor gains
        motor_gains = kwargs.pop('motor_gains', [[0]]*n_joints)

        # Turning body
        for joint_i in range(convention.n_joints_body):
            for side_i in range(2):
                offsets[convention.bodyjoint2index(joint_i=joint_i)] = (
                    AmphibiousMotorOffsetOptions(
                        gain=0,
                        bias=0,
                        low=1,
                        high=5,
                        saturation=0,
                        rate=2,
                    )
                )

        # Turning legs
        legs_offsets_walking = kwargs.pop(
            'legs_offsets_walking',
            [0]*convention.n_dof_legs
        )
        legs_offsets_swimming = kwargs.pop(
            'legs_offsets_swimming',
            [0]*convention.n_dof_legs
        )
        leg_turn_gain = kwargs.pop(
            'leg_turn_gain',
            [0, 0]
            if convention.n_legs == 4
            else (-np.ones(convention.n_legs_pair())).tolist()
        )
        leg_side_turn_gain = kwargs.pop(
            'leg_side_turn_gain',
            [0, 0]
        )
        leg_joint_turn_gain = kwargs.pop(
            'leg_joint_turn_gain',
            [0]*convention.n_dof_legs
        )

        # Augment parameters
        repeat = partial(np.repeat, repeats=convention.n_legs_pair(), axis=0)
        if np.ndim(legs_offsets_walking) == 1:
            legs_offsets_walking = repeat([legs_offsets_walking]).tolist()
        if np.ndim(legs_offsets_swimming) == 1:
            legs_offsets_swimming = repeat([legs_offsets_swimming]).tolist()
        if np.ndim(leg_side_turn_gain) == 1:
            leg_side_turn_gain = repeat([leg_side_turn_gain]).tolist()
        if np.ndim(leg_joint_turn_gain) == 1:
            leg_joint_turn_gain = repeat([leg_joint_turn_gain]).tolist()

        # Motors offsets for walking and swimming
        for leg_i in range(convention.n_legs_pair()):
            for side_i in range(2):
                for joint_i in range(convention.n_dof_legs):
                    offsets[convention.legjoint2index(
                        leg_i=leg_i,
                        side_i=side_i,
                        joint_i=joint_i,
                    )] = AmphibiousMotorOffsetOptions(
                        gain=(
                            leg_turn_gain[leg_i]
                            * leg_side_turn_gain[leg_i][side_i]
                            * leg_joint_turn_gain[leg_i][joint_i]
                        ),
                        bias=legs_offsets_walking[leg_i][joint_i],
                        low=1,
                        high=3,
                        saturation=legs_offsets_swimming[leg_i][joint_i],
                        rate=2,
                    )

        # Amphibious joints control
        if not self.motors:
            self.motors = [
                AmphibiousMotorOptions(
                    joint_name=None,
                    control_types=[],
                    limits_torque=None,
                    gains=None,
                    equation=None,
                    transform=AmphibiousMotorTransformOptions(
                        gain=None,
                        bias=None,
                    ),
                    offsets=AmphibiousMotorOffsetOptions(
                        gain=None,
                        bias=None,
                        low=None,
                        high=None,
                        saturation=None,
                        rate=None,
                    ),
                    passive=AmphibiousPassiveJointOptions(
                        is_passive=False,
                        stiffness_coefficient=0,
                        damping_coefficient=0,
                        friction_coefficient=0,
                    ),
                )
                for joint in range(n_joints)
            ]
        joints_names = kwargs.pop(
            'joints_control_names',
            convention.joints_names,
        )
        transform_gain = kwargs.pop(
            'transform_gain',
            {joint_name: 1 for joint_name in joints_names},
        )
        transform_bias = kwargs.pop(
            'transform_bias',
            {joint_name: 0 for joint_name in joints_names},
        )
        default_max_torque = kwargs.pop('default_max_torque', np.inf)
        max_torques = kwargs.pop(
            'max_torques',
            {joint_name: default_max_torque for joint_name in joints_names},
        )
        default_equation = kwargs.pop('default_equation', 'position')
        equations = kwargs.pop(
            'equations',
            {
                joint_name: (
                    'phase'
                    if convention.single_osc_body
                    and joint_i < convention.n_joints_body
                    or convention.single_osc_legs
                    and joint_i >= convention.n_joints_body
                    else default_equation
                )
                for joint_i, joint_name in enumerate(joints_names)
            },
        )
        for motor_i, motor in enumerate(self.motors):

            # Control
            if motor.joint_name is None:
                motor.joint_name = joints_names[motor_i]
            if motor.equation is None:
                motor.equation = equations[motor.joint_name]
            if not motor.control_types:
                motor.control_types = {
                    'position': ['position'],
                    'phase': ['position'],
                    'ekeberg_muscle': ['velocity', 'torque'],
                    'ekeberg_muscle_explicit': ['torque'],
                    'passive': ['velocity', 'torque'],
                    'passive_explicit': ['torque'],
                }[motor.equation]
            if motor.limits_torque is None:
                motor.limits_torque = [
                    -max_torques[motor.joint_name],
                    +max_torques[motor.joint_name],
                ]
            if motor.gains is None:
                motor.gains = motor_gains[motor_i]

            # Transform
            if motor.transform.gain is None:
                motor.transform.gain = transform_gain[motor.joint_name]
            if motor.transform.bias is None:
                motor.transform.bias = transform_bias[motor.joint_name]

            # Offset
            if motor.offsets.gain is None:
                motor.offsets.gain = offsets[motor_i]['gain']
            if motor.offsets.bias is None:
                motor.offsets.bias = offsets[motor_i]['bias']
            if motor.offsets.low is None:
                motor.offsets.low = offsets[motor_i]['low']
            if motor.offsets.high is None:
                motor.offsets.high = offsets[motor_i]['high']
            if motor.offsets.saturation is None:
                motor.offsets.saturation = offsets[motor_i]['saturation']
            if motor.offsets.rate is None:
                motor.offsets.rate = offsets[motor_i]['rate']

        # Passive
        joints_passive = kwargs.pop('joints_passive', [])
        self.sensors.joints += [name for name, *_ in joints_passive]
        self.motors += [
            AmphibiousMotorOptions(
                joint_name=joint_name,
                control_types=['velocity', 'torque'],
                limits_torque=[-default_max_torque, default_max_torque],
                gains=None,
                equation='passive',
                transform=AmphibiousMotorTransformOptions(
                    gain=1,
                    bias=0,
                ),
                offsets=None,
                passive=AmphibiousPassiveJointOptions(
                    is_passive=True,
                    stiffness_coefficient=stiffness,
                    damping_coefficient=damping,
                    friction_coefficient=friction,
                ),
            )
            for joint_name, stiffness, damping, friction in joints_passive
        ]

        # Muscles
        if not self.muscles:
            self.muscles = [
                AmphibiousMuscleSetOptions(
                    joint_name=None,
                    osc1=None,
                    osc2=None,
                    alpha=None,
                    beta=None,
                    gamma=None,
                    delta=None,
                    epsilon=None,
                )
                for joint_i in range(n_joints)
            ]
        default_alpha = kwargs.pop('muscle_alpha', 0)
        default_beta = kwargs.pop('muscle_beta', 0)
        default_gamma = kwargs.pop('muscle_gamma', 0)
        default_delta = kwargs.pop('muscle_delta', 0)
        default_epsilon = kwargs.pop('muscle_epsilon', 0)
        for joint_i, muscle in enumerate(self.muscles):
            if muscle.joint_name is None:
                muscle.joint_name = joints_names[joint_i]
            if muscle.osc1 is None or muscle.osc2 is None:
                osc_idx = convention.osc_indices(joint_i)
                assert osc_idx[0] < len(self.network.oscillators), (
                    f'{joint_i}: '
                    f'{osc_idx[0]} !< {len(self.network.oscillators)}'
                )
                muscle.osc1 = self.network.oscillators[osc_idx[0]].name
                if len(osc_idx) > 1:
                    assert osc_idx[1] < len(self.network.oscillators), (
                        f'{joint_i}: '
                        f'{osc_idx[1]} !< {len(self.network.oscillators)}'
                    )
                    muscle.osc2 = self.network.oscillators[osc_idx[1]].name
            if muscle.alpha is None:
                muscle.alpha = default_alpha
            if muscle.beta is None:
                muscle.beta = default_beta
            if muscle.gamma is None:
                muscle.gamma = default_gamma
            if muscle.delta is None:
                muscle.delta = default_delta
            if muscle.epsilon is None:
                muscle.epsilon = default_epsilon

    def motors_offsets(self):
        """Motors offsets"""
        return [
            {
                key: getattr(motor.offsets, key)
                for key in [
                    'gain', 'bias',
                    'low', 'high',
                    'saturation_low', 'saturation_high',
                ]
            }
            for motor in self.motors
            if motor.offsets is not None
        ]

    def motors_offset_rates(self):
        """Motors rates"""
        return [
            motor.offsets.rate
            for motor in self.motors
            if motor.offsets is not None
        ]

    def motors_transform_gain(self):
        """Motors gain amplitudes"""
        return [motor.transform.gain for motor in self.motors]

    def motors_transform_bias(self):
        """Motors offset bias"""
        return [motor.transform.bias for motor in self.motors]


class NeuralNetworkOptions(Options):
    """Neural network options"""

    def __init__(self, **kwargs):
        super().__init__()

        # Oscillators
        self.oscillators: List[NeuralOscillatorOptions] = [
            NeuralOscillatorOptions(**oscillator)
            for oscillator in kwargs.pop('oscillators')
        ]
        self.single_osc_body: bool = kwargs.pop('single_osc_body', False)
        self.single_osc_legs: bool = kwargs.pop('single_osc_legs', False)

        # Kwargs
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['oscillators'] = kwargs.pop('oscillators', [])
        options['single_osc_body'] = kwargs.pop('single_osc_body', False)
        options['single_osc_legs'] = kwargs.pop('single_osc_legs', False)
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""

        # Oscillators
        n_oscillators = convention.n_osc()
        if not self.oscillators:
            self.oscillators = [
                NeuralOscillatorOptions(name=None)
                for osc_i in range(n_oscillators)
            ]

    def n_oscillators(self):
        """Number of oscillators"""
        return len(self.oscillators)

    def osc_names(self):
        """Oscillator names"""
        return [osc.name for osc in self.oscillators]

    @staticmethod
    def default_state_init(convention):
        """Default state"""
        state = np.concatenate(
            [
                # Joints inputs
                np.zeros(convention.n_osc()),
                # Joints offsets
                np.zeros(convention.n_joints()),
            ]
        )
        return state


class NeuralOscillatorOptions(Options):
    """Neural oscillator options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name = kwargs.pop('name')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

