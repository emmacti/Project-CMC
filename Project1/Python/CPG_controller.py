import numpy as np
from typing import List
from scipy.integrate import ode

from farms_core.model.data import AnimatData
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.options import AnimatOptions

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.control.network import AnimatNetwork

from farms_core import pylog

from cmc_controllers.polymander_controller import PolymanderController, NeuralNetwork


class CPGNetwork(NeuralNetwork):
    """Dummy Network"""

    def __init__(
        self,
        data: AnimatData,
        drive_left: float,
        drive_right: float,
        d_low: float,
        d_high: float,
        a_rate: np.ndarray,  # (n_joints,)
        offset_freq: np.ndarray,  # (n_joints,)
        offset_amp: np.ndarray,  # (n_joints,)
        G_freq: np.ndarray,  # (n_joints,)
        G_amp: np.ndarray,  # (n_joints,)
        PL: np.ndarray,  # (n_joints,)
        coupling_weights_rostral: float,
        coupling_weights_caudal: float,
        coupling_weights_contra: float,
        init_phase: np.ndarray,  # (2*n_joints,)
        n_body_joints: int,
        left_body_idx: slice,
        right_body_idx: slice,
        **kwargs,
    ):
        super().__init__(data, **kwargs)

        # indexes
        self.n_body_joints = n_body_joints
        self.left_body_idx = left_body_idx
        self.right_body_idx = right_body_idx

        # Controller state
        self.n_body_joints = n_body_joints
        self.n_oscillators = 2*n_body_joints  # double chain
        # [phases, amplitudes, motor_outputs_storage]
        self.state = np.zeros((self.n_iterations, 3*self.n_oscillators))

        # init phase
        self.state[0, :self.n_oscillators] = init_phase

        # Per-oscillator parameter vectors.
        # Oscillator index convention:
        #   oscillator 2*k: left muscle/extensor oscillator for body joint k
        #   oscillator 2*k+1: right muscle/flexor oscillator for body joint k
        self.a_rate_full = np.empty(self.n_oscillators, dtype=float)
        self.a_rate_full[0::2] = np.asarray(a_rate, dtype=float)
        self.a_rate_full[1::2] = np.asarray(a_rate, dtype=float)

        self.offset_freq_full = np.empty(self.n_oscillators, dtype=float)
        self.offset_freq_full[0::2] = np.asarray(offset_freq, dtype=float)
        self.offset_freq_full[1::2] = np.asarray(offset_freq, dtype=float)

        self.offset_amp_full = np.empty(self.n_oscillators, dtype=float)
        self.offset_amp_full[0::2] = np.asarray(offset_amp, dtype=float)
        self.offset_amp_full[1::2] = np.asarray(offset_amp, dtype=float)

        self.G_freq_full = np.empty(self.n_oscillators, dtype=float)
        self.G_freq_full[0::2] = np.asarray(G_freq, dtype=float)
        self.G_freq_full[1::2] = np.asarray(G_freq, dtype=float)

        self.G_amp_full = np.empty(self.n_oscillators, dtype=float)
        self.G_amp_full[0::2] = np.asarray(G_amp, dtype=float)
        self.G_amp_full[1::2] = np.asarray(G_amp, dtype=float)

        # Solver
        self.solver = ode(f=self.network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state[0], t=0.0)

        # CPG controller hyperparameters
        self.d_low = d_low
        self.d_high = d_high
        self.a_rate = a_rate
        self.offset_freq = offset_freq
        self.offset_amp = offset_amp
        self.G_freq = G_freq
        self.G_amp = G_amp
        self.PL = PL
        self.coupling_weights_rostral = coupling_weights_rostral
        self.coupling_weights_caudal = coupling_weights_caudal
        self.coupling_weights_contra = coupling_weights_contra

        pylog.warning("TODO 3.1 stretch feedback")
        self.w_ipsi = kwargs.pop('w_ipsi', None)

        pylog.warning("TODO 3.3 Disruption masks")
        self.disruption_p_sensors = kwargs.pop('disruption_p_sensors', 0.0)
        self.disruption_p_couplings = kwargs.pop('disruption_p_couplings', 0.0)
        self.random_seed = kwargs.pop('random_seed', 42)
        np.random.seed(self.random_seed)

        # CPG controller parameters
        self.nominal_amplitudes = np.zeros(self.n_oscillators, dtype=float)
        self.nominal_frequencies = np.zeros(self.n_oscillators, dtype=float)
        self.coupling_weights = np.zeros(
            (self.n_oscillators, self.n_oscillators), dtype=float
        )
        self.phase_bias = np.zeros(
            (self.n_oscillators, self.n_oscillators), dtype=float
        )

        # Compute nominal frequencies/amplitudes once (Project 1 drive is constant).
        drive_full = np.empty(self.n_oscillators, dtype=float)
        drive_full[0::2] = float(drive_left)
        drive_full[1::2] = float(drive_right)

        in_range = (drive_full >= self.d_low) & (drive_full <= self.d_high)
        self.nominal_frequencies[in_range] = (
            self.G_freq_full[in_range] * (drive_full[in_range] - self.d_low)
            + self.offset_freq_full[in_range]
        )
        self.nominal_amplitudes[in_range] = (
            self.G_amp_full[in_range] * (drive_full[in_range] - self.d_low)
            + self.offset_amp_full[in_range]
        )

        # Initialize oscillator amplitudes at their nominal values.
        self.state[0, self.n_oscillators:2*self.n_oscillators] = self.nominal_amplitudes

        # Phase biases for ipsilateral couplings.
        # PL can be given as:
        #   - scalar (per-adjacent-pair value)
        #   - array length n_body_joints-1 (preferred)
        #   - legacy array length n_body_joints (we take first n_body_joints-1)
        pb = np.asarray(self.PL, dtype=float)
        if pb.ndim == 0:
            pb_adj = np.full(self.n_body_joints - 1, float(pb))
        else:
            pb_flat = pb.reshape(-1)
            if len(pb_flat) == self.n_body_joints - 1:
                pb_adj = pb_flat
            elif len(pb_flat) == self.n_body_joints:
                pb_adj = pb_flat[:-1]
            else:
                raise ValueError(
                    f"PL must be scalar or length (n_body_joints-1={self.n_body_joints-1}) "
                    f"(or legacy length n_body_joints={self.n_body_joints}), got length {len(pb_flat)}."
                )

        # Ipsilateral couplings (adjacent joints) on left (even indices) and right (odd indices).
        for k in range(self.n_body_joints - 1):
            # Left chain
            i_rostral = 2 * k
            j_caudal = 2 * (k + 1)
            self.coupling_weights[i_rostral, j_caudal] = self.coupling_weights_rostral
            self.phase_bias[i_rostral, j_caudal] = +pb_adj[k]
            self.coupling_weights[j_caudal, i_rostral] = self.coupling_weights_caudal
            self.phase_bias[j_caudal, i_rostral] = -pb_adj[k]

            # Right chain
            i_rostral = 2 * k + 1
            j_caudal = 2 * (k + 1) + 1
            self.coupling_weights[i_rostral, j_caudal] = self.coupling_weights_rostral
            self.phase_bias[i_rostral, j_caudal] = +pb_adj[k]
            self.coupling_weights[j_caudal, i_rostral] = self.coupling_weights_caudal
            self.phase_bias[j_caudal, i_rostral] = -pb_adj[k]

        # Contralateral coupling between left and right oscillators at each body joint.
        # From left to right corresponds to +pi (phase of right lags by +pi).
        for k in range(self.n_body_joints):
            left_idx = 2 * k
            right_idx = 2 * k + 1
            self.coupling_weights[left_idx, right_idx] = self.coupling_weights_contra
            self.phase_bias[left_idx, right_idx] = np.pi
            self.coupling_weights[right_idx, left_idx] = self.coupling_weights_contra
            self.phase_bias[right_idx, left_idx] = -np.pi

        # drive (constant in project 1)
        self.drive_left = drive_left
        self.drive_right = drive_right

    def motor_output(self, phase, amplitude):
        # Eq. output_body: M_i = r_i (1 + cos(theta_i))
        oscillator_output = amplitude * (1.0 + np.cos(phase))
        left = np.asarray(oscillator_output[self.left_body_idx], dtype=float)
        right = np.asarray(oscillator_output[self.right_body_idx], dtype=float)
        return left, right

    def network_ode(self, _time, state, stretch_value):
        """
        Compute derivatives for the ODE system.
        state: [phases, amplitudes, dphases_storage, damplitudes_storage, motor_outputs_storage]
        stretch_value: array of stretch feedback values (or zeros if w_ipsi is None)
        Returns: derivatives for [phases, amplitudes]
        """
        phases = state[:self.n_oscillators]
        amplitudes = state[self.n_oscillators:2*self.n_oscillators]

        dstates = np.zeros_like(state)

        # Stretch feedback term is optional (enabled only if w_ipsi is provided).
        # When enabled (and in later exercises), we compute s_i from the current
        # mechanical joint angles theta_i (stretch_value).
        if self.w_ipsi is None or float(self.w_ipsi) == 0.0:
            s = np.zeros(self.n_oscillators, dtype=float)
        else:
            stretch_value = np.asarray(stretch_value, dtype=float).reshape(-1)
            if len(stretch_value) != self.n_body_joints:
                raise ValueError(
                    f"stretch_value must have length n_body_joints={self.n_body_joints}, got {len(stretch_value)}"
                )

            # Eq. v(theta):
            #   theta(left)  = max(0, theta)
            #   theta(right) = max(0, -theta)
            s_left = self.w_ipsi * np.maximum(0.0, stretch_value)
            s_right = self.w_ipsi * np.maximum(0.0, -stretch_value)

            # Interleave to match oscillator indexing (even=left, odd=right).
            s = np.empty(self.n_oscillators, dtype=float)
            s[0::2] = s_left
            s[1::2] = s_right

        eps = 1e-9

        # Amplitude dynamics (Eq. dr with stretch term)
        r_dot = self.a_rate_full * (self.nominal_amplitudes - amplitudes) + s * np.cos(phases)

        # Phase dynamics (Eq. dtheta)
        # Coupling: sum_j r_j * w_ij * sin(theta_j - theta_i - phi_ij)
        phase_diff = phases[None, :] - phases[:, None] - self.phase_bias
        coupling = np.sum(
            (amplitudes[None, :] * self.coupling_weights) * np.sin(phase_diff),
            axis=1,
        )

        # Stretch feedback phase correction: -(s_i/r_i) sin(theta_i)
        stretch_phase_term = np.where(
            amplitudes > eps,
            (s / amplitudes) * np.sin(phases),
            0.0,
        )

        theta_dot = 2.0 * np.pi * self.nominal_frequencies + coupling - stretch_phase_term

        dstates[:self.n_oscillators] = theta_dot
        dstates[self.n_oscillators:2*self.n_oscillators] = r_dot
        return dstates

    def step(
        self,
        iteration: int,
        time: float,
        timestep: float,
        checks: bool = False,
        strict: bool = False,
    ):
        """
        Control step
        Called after obtaining all the current sensor data, and right before
        calling the physics.
        """

        phases = self.state[iteration, :self.n_oscillators]
        amplitudes = self.state[iteration,
                                self.n_oscillators:2*self.n_oscillators]

        # Compute stretch feedback value
        stretch_value = np.array(
            self.data.sensors.joints.array[iteration-1, :self.n_body_joints, 0]) if iteration > 0 else np.zeros(self.n_body_joints)

        # Optional parameters for network_ode (used only if w_ipsi is enabled).
        self.solver.set_f_params(stretch_value)

        # Integrate ODE using dopri5 solver
        self.solver.integrate(time + timestep)
        integrated_state = self.solver.y

        # motor output from CPG state
        motor_output_left, motor_output_right = self.motor_output(
            phases, amplitudes)

        # Only set body joints in project 1
        # self.data.state.array[iteration, :] = 0
        self.data.state.array[iteration,
                              self.left_body_idx] = motor_output_left.copy()
        self.data.state.array[iteration,
                              self.right_body_idx] = motor_output_right.copy()

        # Controller state update
        left_storage_idx = slice(
            self.left_body_idx.start + self.n_oscillators*2,
            self.left_body_idx.stop + self.n_oscillators*2,
            self.left_body_idx.step)
        right_storage_idx = slice(
            self.right_body_idx.start + self.n_oscillators*2,
            self.right_body_idx.stop + self.n_oscillators*2,
            self.right_body_idx.step)
        self.state[iteration, left_storage_idx] = motor_output_left
        self.state[iteration, right_storage_idx] = motor_output_right

        if iteration + 1 >= self.n_iterations:
            return

        # Update state with integrated values
        self.state[iteration+1,
                   :self.n_oscillators] = integrated_state[:self.n_oscillators]
        self.state[iteration +
                   1, self.n_oscillators:2 *
                   self.n_oscillators] = integrated_state[self.n_oscillators:2 *
                                                          self.n_oscillators]


class CPGController(PolymanderController):
    """CPGController"""

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

        animat_network = CPGNetwork(
            data=animat_data,
            drive_left=config['drive_left'],
            drive_right=config['drive_right'],
            d_low=config['d_low'],
            d_high=config['d_high'],
            a_rate=config['a_rate'],
            offset_freq=config['offset_freq'],
            offset_amp=config['offset_amp'],
            G_freq=config['G_freq'],
            G_amp=config['G_amp'],
            PL=config['PL'],
            coupling_weights_rostral=config['coupling_weights_rostral'],
            coupling_weights_caudal=config['coupling_weights_caudal'],
            coupling_weights_contra=config['coupling_weights_contra'],
            init_phase=config['init_phase'],
            w_ipsi=config.get(
                'w_ipsi',
                None),
            disruption_p_sensors=config.get(
                'disruption_p_sensors',
                0.0),
            disruption_p_couplings=config.get(
                'disruption_p_couplings',
                0.0),
            random_seed=config.get(
                'random_seed',
                42),
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

