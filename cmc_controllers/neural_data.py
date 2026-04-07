"""Amphibious data"""

from typing import Dict

import numpy as np

from farms_core import pylog
from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.array.array import to_array
from farms_core.array.array_cy import IntegerArray1D, IntegerArray2D
from farms_core.array.types import NDARRAY_V1
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions, ControlOptions
from farms_core.simulation.options import SimulationOptions
from farms_core.sensors.data import SensorsData
from farms_core.simulation.data import SimulationData
from farms_core.experiment.data import ExperimentData

from farms_amphibious.data.data import (
    AmphibiousKinematicsData,
    OscillatorNetworkState,
)
from farms_amphibious.model.options import (
    AmphibiousControlOptions,
    KinematicsControlOptions,
)


def get_neural_data(animat_options, simulation_options):
    """Get neural data"""
    return NeuralData.from_options(
        animat_options=animat_options,
        simulation_options=simulation_options,
    )


class NeuralOscillators():
    """Oscillator array"""

    def __init__(
            self,
            names: list[str],
    ):
        self.names = names
        self.n_oscillators = len(names)

    @classmethod
    def from_options(cls, network):
        """Default"""
        return cls(network.osc_names())

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(names=dictionary['names'])

    def to_dict(self, iteration: int | None = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {'names': self.names}


class NeuralOscillatorNetworkState(OscillatorNetworkState):
    """Network state"""

    # States
    def phases(self, iteration):
        """Oscillators phases"""
        raise ValueError('Undefined for neural controller')

    def phases_all(self):
        """Oscillators phases"""
        raise ValueError('Undefined for neural controller')

    def amplitudes(self, iteration):
        """Amplitudes"""
        raise ValueError('Undefined for neural controller')

    def amplitudes_all(self):
        """Amplitudes"""
        raise ValueError('Undefined for neural controller')

    def offsets(self, iteration):
        """Offset"""
        return self.array[iteration, self.n_oscillators:]

    def offsets_all(self):
        """Offset"""
        return self.array[:, self.n_oscillators:]

    def outputs(self, iteration):
        """Outputs"""
        return np.array(self.array[iteration, :self.n_oscillators])

    def outputs_all(self):
        """Outputs"""
        return np.array(self.array[:, :self.n_oscillators])

    # Plots
    def plot(self, times) -> Dict:
        """Plot"""
        return {
            'neural_activity': (self.plot_neural_activity(times)),
        }

    def plot_phases(self, times):
        """Plot phases"""
        raise ValueError('No phases to plot')

    def plot_amplitudes(self, times):
        """Plot amplitudes"""
        raise ValueError('No amplitudes to plot')

    def plot_neural_activity(self, times):
        """Plot amplitudes"""
        raise NotImplementedError()


class NeuralNetworkParameters():
    """Network parameter"""

    def __init__(
            self,
            oscillators: NeuralOscillators,
    ):
        self.oscillators = oscillators

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            oscillators=NeuralOscillators.from_dict(
                dictionary['oscillators']
            ),
        ) if dictionary else None

    def to_dict(self, iteration: int | None = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'oscillators': self.oscillators.to_dict(),
        }


class NeuralData(AnimatData):
    """Neural data"""

    def __init__(
            self,
            state: NeuralOscillatorNetworkState,
            network: NeuralNetworkParameters,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.state = state
        self.network = network

    @classmethod
    def from_options(
            cls,
            animat_options: AnimatOptions,
            simulation_options: SimulationOptions,
    ):
        """From animat and simulation options"""

        # Sensors
        sensors = SensorsData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )

        # State
        state = (
            NeuralOscillatorNetworkState.from_initial_state(
                initial_state=animat_options.state_init(),
                n_iterations=simulation_options.runtime.n_iterations,
                n_oscillators=animat_options.control.network.n_oscillators(),
            )
            if animat_options.control.network is not None
            else None
        )

        # Oscillators
        oscillators = NeuralOscillators.from_options(
            network=animat_options.control.network,
        ) if animat_options.control.network is not None else None

        # Network
        network = (
            NeuralNetworkParameters(oscillators=oscillators)
            if animat_options.control.network is not None
            else None
        )

        return cls(
            sensors=sensors,
            state=state,
            network=network,
        )

    @classmethod
    def from_file(cls, filename: str):
        """From file"""
        pylog.info('Loading data from %s', filename)
        data = hdf5_to_dict(filename=filename)
        pylog.info('loaded data from %s', filename)
        data['n_oscillators'] = len(data['network']['oscillators']['names'])
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        n_oscillators = dictionary.pop('n_oscillators')
        return cls(
            state=NeuralOscillatorNetworkState(dictionary['state'], n_oscillators),
            network=NeuralNetworkParameters.from_dict(dictionary['network']),
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        data_dict = super().to_dict(iteration=iteration)
        data_dict.update({
            'state': to_array(self.state.array),
            'network': self.network.to_dict(iteration),
        })
        return data_dict

