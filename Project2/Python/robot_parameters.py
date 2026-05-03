"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super().__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints        # 8
        self.n_legs_joints = parameters.n_legs_joints        # 8 (2 joints x 4 limbs)
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints  # 16
        self.n_oscillators_body = 2 * self.n_body_joints     # 16 (left + right chain)
        self.n_oscillators_legs = 2 * self.n_legs_joints     # 16 (2 antagonists per joint)
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs  # 32

        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([self.n_oscillators, self.n_oscillators])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)
        self.set_coupling_weights(parameters)
        self.set_phase_bias(parameters)
        self.set_amplitudes_rate(parameters)
        self.set_nominal_amplitudes(parameters)

    def step(self, time, iteration, salamandra_data):
        """Step function called at each iteration"""
        gps = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )

    # Drive helpers

    @staticmethod
    def _drive_scalar(parameters):
        """Return drive as a scalar (use current index if array)."""
        d = getattr(parameters, 'drive', 0)
        if np.isscalar(d):
            return float(d)
        # If drive is an array, the caller should have set the current value
        return float(np.atleast_1d(d).flat[0])

    # -------------------------------------------------------------------------
    # Oscillator index helpers
    # -------------------------------------------------------------------------
    # Body oscillators (axial, indices 0–15)
    #   Left  body: 0, 2, 4, 6, 8, 10, 12, 14  (even)
    #   Right body: 1, 3, 5, 7, 9, 11, 13, 15  (odd)
    #
    # Limb oscillators (indices 16–31) — layout:
    #   Each limb has 2 joints (girdle=proximal, knee/elbow=distal).
    #   Each joint uses 2 antagonist oscillators (flexor & extensor).
    #   4 oscillators per limb × 4 limbs = 16 oscillators.
    #
    #   Limb 0 (FL, front-left):   16 (girdle flex), 17 (girdle ext),
    #                               18 (knee flex),   19 (knee ext)
    #   Limb 1 (FR, front-right):  20 (girdle flex), 21 (girdle ext),
    #                               22 (knee flex),   23 (knee ext)
    #   Limb 2 (HL, hind-left):    24 (girdle flex), 25 (girdle ext),
    #                               26 (knee flex),   27 (knee ext)
    #   Limb 3 (HR, hind-right):   28 (girdle flex), 29 (girdle ext),
    #                               30 (knee flex),   31 (knee ext)
    #
    # Within each limb the antagonist pair is in antiphase (π apart).
    # The knee/elbow joint lags the girdle joint by π/2 for circular motion.
    # Left and right limbs are in antiphase (diagonal gait: FL+HR, FR+HL).
    # Forelimb–hindlimb ipsilateral phase lag ≈ π (trot).

    N_BODY = 16   # body oscillator count
    N_LIMBS = 4
    N_OSC_PER_LIMB = 4  # 2 joints × 2 antagonists

    def _limb_base(self, limb_idx):
        """First oscillator index for a given limb (0–3)."""
        return self.N_BODY + limb_idx * self.N_OSC_PER_LIMB

    # 1. Frequencies

    def set_frequencies(self, parameters):
        """Set oscillator intrinsic frequencies driven by MLR drive.

        Parameters follow Ijspeert 2007 supplementary (Table S1):
          Body:  f = cf1_body * d + cf0_body   for d_low_body < d < d_high_body
          Limb:  f = cf1_limb * d + cf0_limb   for d_low_limb < d < d_high_limb
        Outside the active range the frequency is clamped to the boundary value
        (walk below d_low stays at walk freq; swim above d_high stays at swim).
        """
        d = self._drive_scalar(parameters)

        # --- Body oscillators (Ijspeert 2007 Table S1) ---
        # Walking regime:  1.0 < d < 3.0  →  f_body = 0.2*d + 0.3  [Hz]
        # Swimming regime: 3.0 < d < 5.0  →  f_body = 0.1*d + 0.5  [Hz]
        d_low_body  = 1.0
        d_sat_body  = 3.0   # transition walk→swim
        d_high_body = 5.0

        if d < d_low_body:
            f_body = 0.0
        elif d < d_sat_body:
            f_body = 0.2 * d + 0.3     # walking
        elif d <= d_high_body:
            f_body = 0.1 * d + 0.5     # swimming
        else:
            f_body = 0.1 * d_high_body + 0.5   # saturate

        # --- Limb oscillators ---
        # Active only in walking:  1.0 < d < 3.0  →  f_limb = 0.2*d + 0.0  [Hz]
        d_low_limb  = 1.0
        d_high_limb = 3.0

        if d < d_low_limb:
            f_limb = 0.0
        elif d <= d_high_limb:
            f_limb = 0.2 * d + 0.0     # Ijspeert 2007 Table S1
        else:
            f_limb = 0.2 * d_high_limb  # saturate (limbs stop in pure swim)

        self.freqs[:self.N_BODY] = f_body
        self.freqs[self.N_BODY:] = f_limb

    # 2. Coupling weights

    def set_coupling_weights(self, parameters):
        """Set coupling weights w_ij.

        Body chain (Ijspeert 2007):
          - Nearest-neighbour along each side: w = 10
          - Contralateral (left↔right same segment): w = 10

        Limb circuits:
          - Antagonist pair within a joint (flex↔ext): w = 10
          - Girdle → knee coupling (ipsilateral within limb): w = 10
          - Limb → body coupling at corresponding segment: w = 30
          - Body → limb coupling: w = 30
          - Contralateral limb coupling (left↔right): w = 10
          - Fore–hind ipsilateral coupling: w = 10
        """
        W = self.coupling_weights
        W[:] = 0.0

        # ---- Body chain ----
        w_body_chain = 10.0
        w_body_contra = 10.0

        for i in range(self.n_body_joints):
            L = 2 * i        # left oscillator index
            R = 2 * i + 1   # right oscillator index

            # Contralateral coupling (same segment, L↔R)
            W[L, R] = w_body_contra
            W[R, L] = w_body_contra

            # Ipsilateral forward coupling (L→L+2, R→R+2)
            if i + 1 < self.n_body_joints:
                L_next = 2 * (i + 1)
                R_next = 2 * (i + 1) + 1
                W[L, L_next] = w_body_chain
                W[L_next, L] = w_body_chain
                W[R, R_next] = w_body_chain
                W[R_next, R] = w_body_chain

        # ---- Limb circuits ----
        w_antag    = 10.0   # antagonist pair (flexor ↔ extensor)
        w_jnt      = 10.0   # girdle → knee within limb
        w_limb_body = 30.0  # limb ↔ body at corresponding segment
        w_contra_limb = 10.0  # left ↔ right limb (same girdle level)
        w_fore_hind  = 10.0   # fore ↔ hind ipsilateral

        # Forelimb girdles attach near body segment 1 (index 1 = joint 01)
        # Hindlimb girdles attach near body segment 4 (index 4 = joint 04/05)
        # Using body joint indices (0-based): forelimb ≈ joint 1, hindlimb ≈ joint 4
        body_seg_fore = 1   # body joint index for forelimb coupling
        body_seg_hind = 4   # body joint index for hindlimb coupling

        for limb_idx in range(self.N_LIMBS):
            b = self._limb_base(limb_idx)
            # Oscillators: b=girdle_flex, b+1=girdle_ext, b+2=knee_flex, b+3=knee_ext

            # (a) Antagonist coupling: flex ↔ ext for girdle joint
            W[b,   b+1] = w_antag
            W[b+1, b  ] = w_antag
            # (b) Antagonist coupling: flex ↔ ext for knee joint
            W[b+2, b+3] = w_antag
            W[b+3, b+2] = w_antag
            # (c) Girdle → knee coupling (ipsilateral within limb)
            W[b,   b+2] = w_jnt
            W[b+2, b  ] = w_jnt
            W[b+1, b+3] = w_jnt
            W[b+3, b+1] = w_jnt

        # Limb ↔ body coupling
        # Forelimbs: limb 0 (FL), limb 1 (FR)
        for limb_idx, body_seg in [(0, body_seg_fore), (1, body_seg_fore),
                                    (2, body_seg_hind), (3, body_seg_hind)]:
            b = self._limb_base(limb_idx)
            side = limb_idx % 2  # 0=left, 1=right
            body_osc = 2 * body_seg + side  # L or R body oscillator

            W[body_osc, b  ] = w_limb_body
            W[body_osc, b+1] = w_limb_body
            W[b,   body_osc] = w_limb_body
            W[b+1, body_osc] = w_limb_body

        # Contralateral limb coupling (FL↔FR, HL↔HR)
        for pair in [(0, 1), (2, 3)]:
            b0 = self._limb_base(pair[0])
            b1 = self._limb_base(pair[1])
            # girdle oscillators
            W[b0, b1] = w_contra_limb;  W[b1, b0] = w_contra_limb
            W[b0+1, b1+1] = w_contra_limb;  W[b1+1, b0+1] = w_contra_limb

        # Fore–hind ipsilateral coupling (FL↔HL left, FR↔HR right)
        for pair in [(0, 2), (1, 3)]:
            b0 = self._limb_base(pair[0])
            b1 = self._limb_base(pair[1])
            W[b0, b1] = w_fore_hind;  W[b1, b0] = w_fore_hind
            W[b0+1, b1+1] = w_fore_hind;  W[b1+1, b0+1] = w_fore_hind

    # 3. Phase bias

    def set_phase_bias(self, parameters):
        """Set nominal phase lags ψ_ij.

        Body chain:
          - Ipsilateral neighbour lag: phase_lag_body (default 2π/8) per joint
          - Contralateral (same segment): π

        Limb circuits:
          - Antagonist pair: π (antiphase)
          - Knee lags girdle by π/2 (for circular motion at the joint)
          - Contralateral limbs (FL↔FR, HL↔HR): π  (trot: diagonals in phase)
          - Fore–hind ipsilateral: π (diagonal gait)
          - Limb ↔ body: 0 (in phase with local body segment)
        """
        PB = self.phase_bias
        PB[:] = 0.0

        # Body phase lag per segment (default from Project 1: 2π/n_body_joints)
        phase_lag = parameters.phase_lag_body
        if phase_lag is None:
            phase_lag = 2 * np.pi / self.n_body_joints  # travelling wave

        # ---- Body chain ----
        for i in range(self.n_body_joints):
            L = 2 * i
            R = 2 * i + 1

            # Contralateral: π
            PB[L, R] = np.pi
            PB[R, L] = np.pi

            if i + 1 < self.n_body_joints:
                L_next = 2 * (i + 1)
                R_next = 2 * (i + 1) + 1
                # Forward coupling: osc_i leads osc_{i+1} by phase_lag
                PB[L, L_next] = phase_lag
                PB[L_next, L] = -phase_lag
                PB[R, R_next] = phase_lag
                PB[R_next, R] = -phase_lag

        # ---- Limb circuits ----
        for limb_idx in range(self.N_LIMBS):
            b = self._limb_base(limb_idx)
            # b = girdle_flex, b+1 = girdle_ext, b+2 = knee_flex, b+3 = knee_ext

            # Antagonist: antiphase (π)
            PB[b,   b+1] = np.pi;   PB[b+1, b  ] = np.pi
            PB[b+2, b+3] = np.pi;   PB[b+3, b+2] = np.pi

            # Knee lags girdle by π/2
            PB[b,   b+2] = np.pi/2;  PB[b+2, b  ] = -np.pi/2
            PB[b+1, b+3] = np.pi/2;  PB[b+3, b+1] = -np.pi/2

        # Contralateral limb pairs: π (trot)
        for pair in [(0, 1), (2, 3)]:
            b0 = self._limb_base(pair[0])
            b1 = self._limb_base(pair[1])
            PB[b0, b1] = np.pi;  PB[b1, b0] = np.pi
            PB[b0+1, b1+1] = np.pi;  PB[b1+1, b0+1] = np.pi

        # Fore–hind ipsilateral: π (diagonal trot pattern)
        for pair in [(0, 2), (1, 3)]:
            b0 = self._limb_base(pair[0])
            b1 = self._limb_base(pair[1])
            PB[b0, b1] = np.pi;  PB[b1, b0] = np.pi
            PB[b0+1, b1+1] = np.pi;  PB[b1+1, b0+1] = np.pi

        # Limb ↔ body: 0 (no bias — already zero by default)

    # 4. Amplitude convergence rates

    def set_amplitudes_rate(self, parameters):
        """Set convergence rates a_i (same for body and limb, Ijspeert 2007)."""
        # Ijspeert 2007 Table S1: a = 20 for all oscillators
        self.rates[:] = 20.0

    # 5. Nominal amplitudes

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes R_i driven by MLR drive.

        Body (Ijspeert 2007 Table S1):
          Walking:  R_body = 0.065 * d + 0.196   (1 < d < 3)
          Swimming: R_body = 0.131 * d + 0.058   (3 < d < 5)

        Limb:
          Walking:  R_limb = 0.131 * d + 0.131   (1 < d < 3)
          Outside range: 0
        """
        d = self._drive_scalar(parameters)

        # Body
        d_low_body  = 1.0
        d_sat_body  = 3.0
        d_high_body = 5.0

        if d < d_low_body:
            R_body = 0.0
        elif d < d_sat_body:
            R_body = 0.065 * d + 0.196    # walking
        elif d <= d_high_body:
            R_body = 0.131 * d + 0.058    # swimming
        else:
            R_body = 0.131 * d_high_body + 0.058

        # Optional amplitude gradient along the body
        amp_grad = getattr(parameters, 'amplitude_gradient', None)
        if amp_grad is not None:
            # amp_grad is a multiplier array of length n_body_joints
            # applied symmetrically to left and right chains
            for i in range(self.n_body_joints):
                self.nominal_amplitudes[2 * i    ] = R_body * amp_grad[i]
                self.nominal_amplitudes[2 * i + 1] = R_body * amp_grad[i]
        else:
            self.nominal_amplitudes[:self.N_BODY] = R_body

        # Limb
        d_low_limb  = 1.0
        d_high_limb = 3.0

        if d < d_low_limb or d > d_high_limb:
            R_limb = 0.0
        else:
            R_limb = 0.131 * d + 0.131    # Ijspeert 2007 Table S1

        self.nominal_amplitudes[self.N_BODY:] = R_limb