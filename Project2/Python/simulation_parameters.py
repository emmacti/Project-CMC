"""Simulation parameters"""


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 8
        self.duration = 30
        self.timestep = 5e-3
        self.spawn_position = [0, 0, 0.1]
        self.spawn_orientation = [0, 0, 0]
        self.initial_phases = None
        # self.position_body_gain = 0.6  # default do not change
        # self.position_limb_gain = 1  # default do not change
        self.phase_lag_body = None
        self.amplitude_gradient = None
        self.drive = 2.5

        self.drive_transition_low = 3.0
        self.drive_transition_high = 4.0

        # Axial CPG defaults taken from Project 1 
        self.drive_dlow = 1.0
        self.drive_dhigh = 6.0
        self.cpg_convergence_rate = 3.0  # a_i [1/s]
        self.cpg_freq_offset_hz = 1.0  # c_fi,0 [Hz]
        self.cpg_amp_offset = 0.5  # c_Ri,0
        self.cpg_freq_gain = 0.5  # G_freq
        self.cpg_amp_gain = 0.25  # G_amp
        self.cpg_phi_body_total = 2 * 3.141592653589793  # phi_body_total
        self.w_body_rostral = 5.0
        self.w_body_caudal = 5.0
        self.w_body_contra = 10.0
        self.initial_phase_seed = 42

        # Optional experiment knobs
        self.limb_spine_phase_offset = 0.0  # additional phase offset [rad]
        self.disable_limb_spine_coupling = False
        self.axial_amp_gain = 1.0
        self.limb_amp_gain = 1.0

        # Transition controller (exercise 4)
        self.update_drive = False
        self.drive_walk = 2.5
        self.drive_swim = 4.5
        self.transition_contact_threshold = 0.5
        self.transition_hysteresis = 0.2

        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

