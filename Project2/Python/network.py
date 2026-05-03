"""Oscillator network ODE"""

import numpy as np
# from scipy.integrate import ode
from scipy import integrate
from scipy.integrate._ode import ode as ODE
from robot_parameters import RobotParameters

import farms_core.pylog as pylog
from farms_amphibious.control.network import NetworkODE


def network_ode(_time, state, robot_parameters, loads, contact_sens):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters
    loads: <np.array>
        The lateral forces applied to the body links

    Returns
    -------
    dstate: <np.array>
        Returns derivative of state (phases and amplitudes)

    """
    n_oscillators = robot_parameters.n_oscillators
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]
    phase_repeat = np.repeat(
        np.expand_dims(phases, axis=1),
        n_oscillators,
        axis=1,
    )
    # loads_expanded = np.concatenate([
    #     loads[:robot_parameters.n_body_joints],
    #     loads[:robot_parameters.n_body_joints],
    #     loads[robot_parameters.n_body_joints:],
    #     loads[robot_parameters.n_body_joints:],
    # ])
    dphases = (
        # Intrinsic frequencies
        robot_parameters.freqs
        # Coupling
        + np.sum(
            amplitudes * robot_parameters.coupling_weights.T * np.sin(
                phase_repeat.T - phase_repeat + robot_parameters.phase_bias.T),
            axis=1,
        )
        # Hydrodynamics sensory feedback
        # + robot_parameters.feedback_gains_swim * loads_expanded
    )
    if contact_sens is not None:
        dphases += contact_sens*np.cos(phases)

    damplitudes = robot_parameters.rates*(
        robot_parameters.nominal_amplitudes
        - amplitudes
    )
    return np.concatenate([dphases, damplitudes])


def motor_output(phases, amplitudes, iteration):
    """Motor output

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    motor_outputs: <np.array>
        Motor outputs for joint in the system.

    """
    output = (
        amplitudes[0:32:2]*(1+np.cos(phases[0:32:2]))
        - amplitudes[1:32:2]*(1+np.cos(phases[1:32:2]))
    ) if iteration is not None else (
        amplitudes[:, 0:32:2]*(1+np.cos(phases[:, 0:32:2]))
        - amplitudes[:, 1:32:2]*(1+np.cos(phases[:, 1:32:2]))
    )
    return output


class SalamandraNetwork(NetworkODE):
    """Salamandra oscillator network"""

    def __init__(self, sim_parameters, n_iterations, data):
        super().__init__(data, ode=network_ode)
        self.n_iterations = n_iterations
        # States
        self.state = data.state
        # Parameters
        self.robot_parameters = RobotParameters(sim_parameters)
        # self.drive_config = None
        # Set initial state
        # Replace your initial oscillator phases here if needed
        self.state.set_phases(
            iteration=0,
            value=1e-4*np.random.rand(self.robot_parameters.n_oscillators),
        )
        # Set solver
        self.integrator = 'dopri5'
        self.integrator_kwargs = {}
        self.solver = integrate.ode(f=network_ode)
        self.solver.set_integrator(self.integrator)
        self.solver.set_initial_value(y=self.state.array[0, :64], t=0.0)

    def initialize_episode(self):
        """Initialize episode"""
        self.solver: ODE = integrate.ode(f=network_ode)
        self.solver.set_integrator(self.integrator, **self.integrator_kwargs)
        self.solver.set_initial_value(y=self.data.state.array[0, :64], t=0.0)
        self.data.state.array[1:, :] = 0

    def step(self, iteration, time, timestep, loads=None, contact_sens=None):
        """Step"""
        if loads is None:
            loads = np.zeros(self.robot_parameters.n_joints)
        if iteration + 1 >= self.n_iterations:
            return
        self.solver.set_f_params(self.robot_parameters, loads, contact_sens)
        self.state.array[iteration+1,
                         :64] = self.solver.integrate(time+timestep)

    def outputs(self, iteration=None):
        """Oscillator outputs"""
        amplitudes = self.state.amplitudes(iteration=iteration)
        phases = self.state.phases(iteration=iteration)
        return amplitudes*(1+np.cos(phases))

    def get_motor_activations(self, iteration=None):
        """Get motor position"""
        oscillator_output = motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )
        return oscillator_output

