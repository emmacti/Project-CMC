"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super().__init__()
        self.sim_parameters = parameters

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = 2*self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.phase_bias = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        # self.feedback_gains_swim = np.zeros(self.n_oscillators)
        # self.feedback_gains_walk = np.zeros(self.n_oscillators)

        # # gains for final motor output
        # self.position_body_gain = parameters.position_body_gain
        # self.position_limb_gain = parameters.position_limb_gain

        self.update(parameters)

    @staticmethod
    def _as_drive_scalar(parameters, iteration: int | None = None) -> float:
        """Return scalar drive at a given iteration.

        The project uses a single descending drive, but exercises may pass an
        array-like profile for ramps.
        """
        drive = getattr(parameters, 'drive', 0.0)
        if iteration is None:
            return float(drive) if np.ndim(drive) == 0 else float(np.asarray(drive)[0])
        if np.ndim(drive) == 0:
            return float(drive)
        drive_arr = np.asarray(drive, dtype=float)
        if len(drive_arr) == 0:
            return 0.0
        if iteration < 0:
            iteration = 0
        if iteration >= len(drive_arr):
            return float(drive_arr[-1])
        return float(drive_arr[iteration])

    @staticmethod
    def _ramp(x: float, x0: float, x1: float) -> float:
        """Clamped linear ramp from 0 at x0 to 1 at x1."""
        if x1 == x0:
            return 1.0 if x >= x1 else 0.0
        return float(np.clip((x - x0) / (x1 - x0), 0.0, 1.0))

    def _drive_blend_walk_swim(self, drive: float) -> tuple[float, float]:
        """Return (walk_blend, swim_blend) weights that sum to 1."""
        # Low drive -> walking, high drive -> swimming. Smooth transition.
        d0 = float(getattr(self.sim_parameters, 'drive_transition_low', 3.0))
        d1 = float(getattr(self.sim_parameters, 'drive_transition_high', 4.0))
        swim = self._ramp(drive, d0, d1)
        walk = 1.0 - swim
        return walk, swim

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # psi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def step(self, time, iteration, salamandra_data):
        """Step function called at each iteration

        Parameters
        ----------

        salamanra_data: salamandra_simulation/data.py::SalamandraData
            Contains the robot data, including network and sensors.

        gps (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 representing the GPS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        # Example to get global coordinates of robot links
        gps = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )
        del gps  # used in some strategies

        # Optional drive update (exercise 4)
        if getattr(self.sim_parameters, 'update_drive', False):
            index = 0 if iteration == 0 else (iteration - 1)
            contacts_all = np.linalg.norm(np.array(
                salamandra_data.sensors.contacts.totals()[index]
            ), axis=1)
            contacts_feet = contacts_all[10:18:2]
            feet_contact_level = float(np.mean(contacts_feet))

            thr = float(getattr(self.sim_parameters, 'transition_contact_threshold', 0.5))
            hyst = float(getattr(self.sim_parameters, 'transition_hysteresis', 0.2))
            drive_walk = float(getattr(self.sim_parameters, 'drive_walk', 2.5))
            drive_swim = float(getattr(self.sim_parameters, 'drive_swim', 4.5))

            current_drive = self._as_drive_scalar(self.sim_parameters, iteration=iteration)
            # Land if feet have contact, water otherwise. Hysteresis for stability.
            if current_drive >= drive_swim - 1e-9:
                #  swimming -> switch to walking only with strong contact
                if feet_contact_level > (thr + hyst):
                    self.sim_parameters.drive = drive_walk
            else:
                #  walking -> switch to swimming when no contact 
                if feet_contact_level < max(0.0, thr - hyst):
                    self.sim_parameters.drive = drive_swim

        drive = self._as_drive_scalar(self.sim_parameters, iteration=iteration)
        # Temporarily replace the drive array with the current scalar so that
        # set_frequencies / set_nominal_amplitudes see the correct value.
        # Without this they call _as_drive_scalar(params, iteration=None) which
        # always returns array[0] (= 0.0 for a ramp), silencing the axial CPG.
        _saved_drive = self.sim_parameters.drive
        self.sim_parameters.drive = drive
        self.set_frequencies(self.sim_parameters)
        self.set_nominal_amplitudes(self.sim_parameters)
        self.set_phase_bias(self.sim_parameters)
        self.set_coupling_weights(self.sim_parameters)
        self.sim_parameters.drive = _saved_drive

    def set_frequencies(self, parameters):
        """Set frequencies"""
        # Axial Freq for CPG in Project 1  
        d = self._as_drive_scalar(parameters, iteration=None)
        dlow = float(getattr(parameters, 'drive_dlow', 1.0))
        dhigh = float(getattr(parameters, 'drive_dhigh', 5.0))
        cf0 = float(getattr(parameters, 'cpg_freq_offset_hz', 1.0))
        gfreq = float(getattr(parameters, 'cpg_freq_gain', 0.5))

        if d < dlow or d > dhigh:
            f_body_hz = 0.0
        else:
            f_body_hz = gfreq * (d - dlow) + cf0

        # Network ODE uses 2π f_i, so we store omega = 2π f
        self.freqs[:self.n_oscillators_body] = 2 * np.pi * f_body_hz

        # limbs active in walking-like drives only.
        walk_blend, swim_blend = self._drive_blend_walk_swim(d)
        f_limb_hz = (0.3 + 0.15 * d) * walk_blend + 0.0 * swim_blend
        self.freqs[self.n_oscillators_body:] = 2 * np.pi * f_limb_hz

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        n = self.n_oscillators
        w = np.zeros((n, n), dtype=float)

        # Axial couplings from Project 1 
        w_body_rostral = float(getattr(parameters, 'w_body_rostral', 5.0))
        w_body_caudal = float(getattr(parameters, 'w_body_caudal', 5.0))
        w_body_contra = float(getattr(parameters, 'w_body_contra', 10.0))

        # Limb/spine-limb couplings 
        w_limb_antag = 12.0
        w_limb_intra = 8.0
        w_limb_inter = 6.0
        w_spine_limb = 4.0

        # Body chain coupling: even indices and odd = antagonist chains
        # Coupling neighbors along each chain with directional rostral/caudal weights
        for joint_i in range(self.n_body_joints - 1):
            a0 = 2*joint_i
            a1 = 2*(joint_i+1)
            b0 = 2*joint_i + 1
            b1 = 2*(joint_i+1) + 1
            # reminder : head -> tail = caudal, tail -> head = rostral
            w[a0, a1] = w_body_caudal
            w[a1, a0] = w_body_rostral
            w[b0, b1] = w_body_caudal
            w[b1, b0] = w_body_rostral

        # Contralateral (antagonist) coupling within each joint
        for joint_i in range(self.n_body_joints):
            i0 = 2*joint_i
            i1 = 2*joint_i + 1
            w[i0, i1] = w_body_contra
            w[i1, i0] = w_body_contra

        # Limb oscillator indices by joint
        limb_base = self.n_oscillators_body  # 16
        # limb joints: 8 joints ordered as (LF shoulder, LF elbow, RF shoulder, RF elbow, LH hip, LH knee, RH hip, RH knee)
        # Each joint has 2 antagonists => 2 oscillators
        def limb_joint_osc(joint_k: int) -> tuple[int, int]:
            idx0 = limb_base + 2*joint_k
            return idx0, idx0 + 1

        # Antagonist coupling for each limb joint
        for joint_k in range(self.n_legs_joints):
            i0, i1 = limb_joint_osc(joint_k)
            w[i0, i1] = w_limb_antag
            w[i1, i0] = w_limb_antag

        # Intra-limb coupling between proximal and distal joint (circular movement)
        for limb in range(4):
            prox = 2*limb + 0
            dist = 2*limb + 1
            p0, p1 = limb_joint_osc(prox)
            d0, d1 = limb_joint_osc(dist)
            for a, b in ((p0, d0), (p1, d1)):
                w[a, b] = w_limb_intra
                w[b, a] = w_limb_intra

        # Inter-limb coordination (trot-like): LF <-> RH in-phase, RF <-> LH in-phase, left-right anti-phase
        # Apply to both joints (prox+dist) by coupling corresponding joints.
        lf = [0, 1]
        rf = [2, 3]
        lh = [4, 5]
        rh = [6, 7]
        pairs_in_phase = [(lf, rh), (rf, lh)]
        pairs_anti = [(lf, rf), (lh, rh)]
        for (group_a, group_b) in pairs_in_phase + pairs_anti:
            for joint_a, joint_b in zip(group_a, group_b):
                a0, a1 = limb_joint_osc(joint_a)
                b0, b1 = limb_joint_osc(joint_b)
                for x, y in ((a0, b0), (a1, b1)):
                    w[x, y] = w_limb_inter
                    w[y, x] = w_limb_inter

        # Spine-limb coupling
        if not getattr(parameters, 'disable_limb_spine_coupling', False):
            # attach forelimbs and hindlimbs to correct body joints
            fore_body_joint = 1
            hind_body_joint = 6
            fore_osc = (2*fore_body_joint, 2*fore_body_joint+1)
            hind_osc = (2*hind_body_joint, 2*hind_body_joint+1)
            # Couple both limb proximal joints to respective body joint pair
            for limb_joints, body_pair in (((0, 2), fore_osc), ((4, 6), hind_osc)):
                for limb_prox_joint in limb_joints:
                    l0, l1 = limb_joint_osc(limb_prox_joint)
                    for bo in body_pair:
                        w[bo, l0] = w_spine_limb
                        w[l0, bo] = w_spine_limb
                        w[bo, l1] = w_spine_limb
                        w[l1, bo] = w_spine_limb

        self.coupling_weights = w

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        n = self.n_oscillators
        psi = np.zeros((n, n), dtype=float)

        # Body biases
        phi_body_total = float(getattr(parameters, 'cpg_phi_body_total', 2*np.pi))
        pb = phi_body_total / float(self.n_body_joints)
        phase_lag_body_param = getattr(parameters, 'phase_lag_body', None)
        phase_lag_body = float(pb if phase_lag_body_param is None else phase_lag_body_param)

        for joint_i in range(self.n_body_joints - 1):
            a0 = 2*joint_i
            a1 = 2*(joint_i+1)
            b0 = 2*joint_i + 1
            b1 = 2*(joint_i+1) + 1
            # forward wave from head->tail
            psi[a0, a1] = -phase_lag_body
            psi[a1, a0] = +phase_lag_body
            psi[b0, b1] = -phase_lag_body
            psi[b1, b0] = +phase_lag_body

        # Contralateral phase biases: left->right π, right->left -π
        for joint_i in range(self.n_body_joints):
            i0 = 2*joint_i
            i1 = 2*joint_i + 1
            psi[i0, i1] = np.pi
            psi[i1, i0] = -np.pi

        # Limb biases
        limb_base = self.n_oscillators_body
        limb_spine_offset = float(getattr(parameters, 'limb_spine_phase_offset', 0.0))
        limb_joint_phase = -np.pi/2  # proximal leads distal by 90deg (circular)

        def limb_joint_osc(joint_k: int) -> tuple[int, int]:
            idx0 = limb_base + 2*joint_k
            return idx0, idx0 + 1

        # Antagonists per limb joint
        for joint_k in range(self.n_legs_joints):
            i0, i1 = limb_joint_osc(joint_k)
            psi[i0, i1] = np.pi
            psi[i1, i0] = np.pi

        # Proximal-distal within limb
        for limb in range(4):
            prox = 2*limb + 0
            dist = 2*limb + 1
            p0, p1 = limb_joint_osc(prox)
            d0, d1 = limb_joint_osc(dist)
            for a, b in ((p0, d0), (p1, d1)):
                psi[a, b] = limb_joint_phase
                psi[b, a] = -limb_joint_phase

        # Inter-limb: trot-like
        lf = [0, 1]
        rf = [2, 3]
        lh = [4, 5]
        rh = [6, 7]
        pairs_in_phase = [(lf, rh), (rf, lh)]
        pairs_anti = [(lf, rf), (lh, rh)]
        for (group_a, group_b) in pairs_in_phase:
            for joint_a, joint_b in zip(group_a, group_b):
                a0, a1 = limb_joint_osc(joint_a)
                b0, b1 = limb_joint_osc(joint_b)
                for x, y in ((a0, b0), (a1, b1)):
                    psi[x, y] = 0.0
                    psi[y, x] = 0.0
        for (group_a, group_b) in pairs_anti:
            for joint_a, joint_b in zip(group_a, group_b):
                a0, a1 = limb_joint_osc(joint_a)
                b0, b1 = limb_joint_osc(joint_b)
                for x, y in ((a0, b0), (a1, b1)):
                    psi[x, y] = np.pi
                    psi[y, x] = np.pi

        # Spine-limb coordination phase offset 
        if not getattr(parameters, 'disable_limb_spine_coupling', False):
            fore_body_joint = 1
            hind_body_joint = 6
            fore_osc = (2*fore_body_joint, 2*fore_body_joint+1)
            hind_osc = (2*hind_body_joint, 2*hind_body_joint+1)
            for limb_prox_joint, body_pair in ((0, fore_osc), (2, fore_osc), (4, hind_osc), (6, hind_osc)):
                l0, l1 = limb_joint_osc(limb_prox_joint)
                for bo in body_pair:
                    psi[bo, l0] = limb_spine_offset
                    psi[l0, bo] = -limb_spine_offset
                    psi[bo, l1] = limb_spine_offset
                    psi[l1, bo] = -limb_spine_offset

        self.phase_bias = psi

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates[:] = float(getattr(parameters, 'cpg_convergence_rate', 3.0))

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        d = self._as_drive_scalar(parameters, iteration=None)
        dlow = float(getattr(parameters, 'drive_dlow', 1.0))
        dhigh = float(getattr(parameters, 'drive_dhigh', 5.0))
        cr0 = float(getattr(parameters, 'cpg_amp_offset', 0.5))
        gamp = float(getattr(parameters, 'cpg_amp_gain', 0.25))

        axial_gain = float(getattr(parameters, 'axial_amp_gain', 1.0))
        limb_gain = float(getattr(parameters, 'limb_amp_gain', 1.0))

        if d < dlow or d > dhigh:
            r_body = 0.0
        else:
            r_body = gamp * (d - dlow) + cr0
        r_body *= axial_gain

        self.nominal_amplitudes[:self.n_oscillators_body] = r_body

        walk_blend, swim_blend = self._drive_blend_walk_swim(d)
        r_limb_walk = 1.0 + 0.05 * d
        r_limb_swim = 0.05
        r_limb = limb_gain * (r_limb_walk * walk_blend + r_limb_swim * swim_blend)
        self.nominal_amplitudes[self.n_oscillators_body:] = r_limb

