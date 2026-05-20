"""Animat data"""

import numpy as np

from farms_core import pylog
from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.simulation.options import SimulationOptions
from farms_core.sensors.data import SensorsData
from farms_amphibious.data.data import (
    AmphibiousData,
    OscillatorNetworkState,
    # JointsControlArray,
)
# from farms_amphibious.data.network import (
#     DriveArray,
#     Oscillators,
#     NetworkParameters,
# )

NPDTYPE = np.float64
NPITYPE = np.uintc


# def to_array(array, iteration=None):
#     """To array or None"""
#     if array is not None:
#         array = np.array(array)
#         if iteration is not None:
#             array = array[:iteration]
#     return array


class SalamandraState(OscillatorNetworkState):
    """Salamandra state"""

    # def __init__(self, array):
    #     super().__init__()
    #     self.array = array

    @classmethod
    def salamandra_robot(cls, n_iterations, n_oscillators):
        """State of Salamandra robot"""
        return cls(
            array=np.zeros([n_iterations, 2*32]),
            n_oscillators=n_oscillators,
        )

    def phases(self, iteration=None):
        """Oscillator phases"""
        return (
            self.array[iteration, :32]
            if iteration is not None
            else self.array[:, :32]
        )

    def set_phases(self, iteration, value):
        """Set phases"""
        self.array[iteration, :32] = value

    def set_phases_left(self, iteration, value):
        """Set body phases on left side"""
        self.array[iteration, 0:16:2] = value

    def set_phases_right(self, iteration, value):
        """Set body phases on right side"""
        self.array[iteration, 1:16:2] = value

    def set_phases_legs(self, iteration, value):
        """Set leg phases"""
        self.array[iteration, 16:32] = value

    def amplitudes(self, iteration=None):
        """Oscillator amplitudes"""
        return (
            self.array[iteration, 32:64]
            if iteration is not None
            else self.array[:, 32:64]
        )

    def set_amplitudes(self, iteration, value):
        """Set amplitudes"""
        self.array[iteration, 32:64] = value


# class NeuralOscillators(Oscillators):
#     """Oscillator array"""

#     def __init__(
#             self,
#             names: list[str],
#     ):
#         self.names = names
#         self.n_oscillators = len(names)

#     @classmethod
#     def from_options(cls, network):
#         """Default"""
#         return cls(network.osc_names())

#     @classmethod
#     def from_dict(cls, dictionary: dict):
#         """Load data from dictionary"""
#         return cls(names=dictionary['names'])

#     def to_dict(self, iteration: int | None = None) -> dict:
#         """Convert data to dictionary"""
#         assert iteration is None or isinstance(iteration, int)
#         return {'names': self.names}


# class NeuralNetworkParameters(NetworkParameters):
#     """Network parameter"""

#     def __init__(
#             self,
#             oscillators: NeuralOscillators,
#             drives: DriveArray,
#     ):
#         self.oscillators = oscillators
#         self.drives= drives

#     @classmethod
#     def from_dict(cls, dictionary: dict):
#         """Load data from dictionary"""
#         return cls(
#             oscillators=NeuralOscillators.from_dict(
#                 dictionary['oscillators']
#             ),
#             drives=DriveArray.from_dict(
#                 dictionary['drives']
#             ),
#         ) if dictionary else None

#     def to_dict(self, iteration: int | None = None) -> dict:
#         """Convert data to dictionary"""
#         assert iteration is None or isinstance(iteration, int)
#         return {'oscillators': self.oscillators.to_dict()}


class SalamandraData(AmphibiousData):
    """Salamandra data"""

    # def __init__(
    #         self,
    #         # state: SalamandraState,
    #         state: OscillatorNetworkState,
    #         network: NeuralNetworkParameters,
    #         joints: JointsControlArray,
    #         **kwargs,
    # ):
    #     pylog.debug('Salamandra data being set')
    #     super().__init__(
    #         state=state,
    #         network=network,
    #         joints=joints,
    #         **kwargs,
    #     )
    #     pylog.debug('Salamandra data is set')
    #     # self.state: SalamandraState = state
    #     # self.state = state
    #     # self.network = network

    @classmethod
    def from_options(
            cls,
            animat_options: AnimatOptions,
            simulation_options: SimulationOptions,
    ):
        """From options"""
        data = super().from_options(animat_options, simulation_options)
        data.state = SalamandraState.from_initial_state(
            initial_state=animat_options.state_init(),
            n_iterations=simulation_options.runtime.n_iterations,
            n_oscillators=animat_options.control.network.n_oscillators(),
        )
        return data

        # # Sensors
    #     sensors = SensorsData.from_options(
    #         animat_options=animat_options,
    #         simulation_options=simulation_options,
    #     )

    #     # # State
    #     # state = SalamandraState.salamandra_robot(
    #     #     n_iterations=simulation_options.runtime.n_iterations,
    #     # )
    #     # State
    #     state = (
    #         SalamandraState.from_initial_state(
    #             initial_state=animat_options.state_init(),
    #             n_iterations=simulation_options.runtime.n_iterations,
    #             n_oscillators=animat_options.control.network.n_oscillators(),
    #         )
    #         if animat_options.control.network is not None
    #         else None
    #     )

    #     # # Network
    #     # network = (
    #     #     NetworkParameters(
    #     #         drives=DriveArray.from_animat_options(
    #     #             animat_options=animat_options,
    #     #             n_iterations=simulation_options.runtime.n_iterations,
    #     #         ),
    #     #         oscillators=oscillators,
    #     #         osc2osc_map=OscillatorConnectivity.from_connectivity(
    #     #             connectivity=animat_options.control.network.osc2osc,
    #     #             map1=oscillators_map,
    #     #             map2=oscillators_map,
    #     #         ),
    #     #         joints2osc_map=JointsConnectivity.from_connectivity(
    #     #             connectivity=animat_options.control.network.joint2osc,
    #     #             map1=oscillators_map,
    #     #             map2=joints_map,
    #     #         ),
    #     #         contacts2osc_map=(
    #     #             ContactsConnectivity.from_connectivity(
    #     #                 connectivity=animat_options.control.network.contact2osc,
    #     #                 map1=oscillators_map,
    #     #                 map2=contacts_map,
    #     #             )
    #     #         ),
    #     #         xfrc2osc_map=XfrcConnectivity.from_connectivity(
    #     #             connectivity=animat_options.control.network.xfrc2osc,
    #     #             map1=oscillators_map,
    #     #             map2=xfrc_map,
    #     #         ),
    #     #     )
    #     #     if animat_options.control.network is not None
    #     #     else None
    #     # )

    #     # Oscillators
    #     oscillators = NeuralOscillators.from_options(
    #         network=animat_options.control.network,
    #     ) if animat_options.control.network is not None else None

    #     # Network
    #     network = (
    #         NeuralNetworkParameters(
    #             oscillators=oscillators,
    #             drives=DriveArray.from_animat_options(
    #                 animat_options=animat_options,
    #                 n_iterations=simulation_options.runtime.n_iterations,
    #             ),
    #         )
    #         if animat_options.control.network is not None
    #         else None
    #     )

    #     return cls(
    #         # timestep=simulation_options.physics.timestep,
    #         sensors=sensors,
    #         state=state,
    #         network=network,
    #         joints=(
    #             JointsControlArray.from_options(animat_options.control)
    #             if animat_options.control.network is not None
    #             else None
    #         ),
    #     )

    # # @classmethod
    # # def from_file(cls, filename: str):
    # #     """From file"""
    # #     pylog.info('Loading data from %s', filename)
    # #     data = hdf5_to_dict(filename=filename)
    # #     pylog.info('loaded data from %s', filename)
    # #     return cls.from_dict(data)

    # # @classmethod
    # # def from_dict(cls, dictionary: dict):
    # #     """Load data from dictionary"""
    # #     return cls(
    # #         timestep=dictionary['timestep'],
    # #         sensors=SensorsData.from_dict(dictionary['sensors']),
    # #         state=SalamandraState(array=dictionary['state']),
    # #     )

    # # def to_dict(self, iteration: int = None) -> dict:
    # #     """Convert data to dictionary"""
    # #     data_dict = super().to_dict(iteration=iteration)
    # #     data_dict.update({'state': to_array(self.state.array)})
    # #     return data_dict

    # # def plot(self, times) -> dict:
    # #     """Plot"""
    # #     plots = {}
    # #     plots.update(self.plot_sensors(times))
    # #     return plots

