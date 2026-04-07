import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, fftconvolve, medfilt
from farms_core import pylog

LINKS_MASSES = np.array(
    [
        0.328768, 0.274101, 0.107688, 0.107688, 0.107688,  # link_body_00 - link_body_04
        0.0433459, 0.107688, 0.107688,  # link_body_06 - link_body_08
        0.18959,  # link_body_10
        0.0194482, 0.164364, 0.0194482, 0.164364,  # link_leg_0_L - link_leg_0_R
        0.0194482, 0.164364, 0.0194482, 0.164364,  # link_leg_1_L - link_leg_1_R
        0.321614, 0.164651,  # link_body_05 link_body_09
    ]
)

TOTAL_MASS = np.sum(LINKS_MASSES)

# Signal filtering


def get_filtered_signals(
    signals: np.ndarray,
    signal_dt: float,
    fcut_hp: float = None,
    fcut_lp: float = None,
    filt_order: int = 5,
):
    """Butterworth zero-phase filtering (works for 1D or 2D)."""

    fnyq = 0.5 / signal_dt

    if fcut_hp is not None:
        num, den = butter(filt_order, fcut_hp / fnyq, btype="highpass")
        signals = filtfilt(num, den, signals, axis=0)

    if fcut_lp is not None:
        num, den = butter(filt_order, fcut_lp / fnyq, btype="lowpass")
        signals = filtfilt(num, den, signals, axis=0)

    return signals


def remove_signals_offset(signals: np.ndarray):
    signals = np.asarray(signals)
    if signals.ndim == 1:
        return signals - np.mean(signals)
    elif signals.ndim == 2:
        return signals - np.mean(signals, axis=1, keepdims=True)
    else:
        raise ValueError(
            f"signals must be 1D or 2D, got shape {signals.shape}")


def filter_signals(
    times: np.ndarray,
    signals: np.ndarray,
):

    dt_sig = times[1]-times[0]

    # Filter high-frequency noise
    smooth_signals = signals.copy()
    smooth_signals = remove_signals_offset(smooth_signals)
    smooth_signals = get_filtered_signals(smooth_signals, dt_sig, fcut_lp=50)
    return smooth_signals

# NM1
# Neural frequency and neural amplitude


def compute_frequency_amplitude_fft(
        times: np.ndarray, 
        smooth_signals: np.ndarray):
    """Return dominant frequency index and amplitude per channel (or per 1D signal)."""

    dt_sig = times[1] - times[0]
    n_step = len(times)
    x = np.asarray(smooth_signals)

    if x.ndim == 1:
        x = x[:, None]   # (time, 1)
    elif x.ndim != 2:
        raise ValueError(f"smooth_signals must be 1D or 2D, got {x.shape}")

    # FFT along time
    # zero-pad to next power of 2 for better frequency resolution (for
    # exercise 3.1)
    Nfft = 16*n_step
    fft_vals = np.fft.rfft(x, n=Nfft, axis=0)              # (n_freq, channels)
    freqs = np.fft.rfftfreq(Nfft, d=dt_sig)     # (n_freq,)
    mag = np.abs(fft_vals)                         # (n_freq, channels)

    # ignore DC when searching peak
    peak_idx = np.argmax(mag[1:, :], axis=0) + 1   # (channels)
    peak_freq = freqs[peak_idx]                    # (channels,)

    # amplitude ≈ 2*|FFT|/N for non-DC bins
    peak_amp = 2.0 * mag[peak_idx,
                         np.arange(mag.shape[1])] / n_step  # (channels,)

    # If originally 1D, return scalars
    if smooth_signals.ndim == 1:
        return peak_freq[0], peak_idx[0], peak_amp[0]

    return peak_freq, freqs, peak_amp

# NM2
# Intersegmental phase lag (IPL)


def get_phase_lag(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sig_dt: float,
    sig_freq: float,
):
    ''' Compute the phase lag between two signals in radians'''

    xcorr = np.correlate(sig2, sig1, "full")
    n_lag = np.argmax(xcorr) - len(xcorr) // 2

    # Convert sample lag to time delay, then to phase in radians
    dt_lag = n_lag * sig_dt
    phase_lag = 2.0 * np.pi * sig_freq * dt_lag

    # Wrap to (-pi, pi] for a stable mean across pairs
    phase_lag = (phase_lag + np.pi) % (2.0 * np.pi) - np.pi
    return phase_lag


def compute_neural_phase_lags(
    times: np.ndarray,
    smooth_signals: np.ndarray,
    freqs: np.ndarray,
    inds_couples: list[list[int, int]]
) -> np.ndarray:
    '''
    Computes the intersegmental phase lags (IPLs) between pairs of signals.

    Input:
    times: time array of shape (T,)
    smooth_signals: smoothed neural signals of shape (T, n_signals)
    freqs: dominant frequencies of shape (n_signals,) (Compute this with NM1)
    inds_couples: list of pairs of indices for which to compute IPLs (Consider adjacent pairs)

    Return:
    ipls: Intersegmental phase lag array of shape (n_couples,)
    ipls_mean: Mean intersegmental phase lag across couples (scalar)
    '''
    ipls = np.zeros(len(inds_couples))
    dt_sig = times[1] - times[0]

    for ind_couple, (ind1, ind2) in enumerate(inds_couples):
        ipls[ind_couple] = get_phase_lag(
            sig1=smooth_signals[:, ind1],
            sig2=smooth_signals[:, ind2],
            sig_dt=dt_sig,
            sig_freq=np.mean([freqs[ind1], freqs[ind2]]),
        )

    return ipls, np.mean(ipls)

# MM1
# Mechanical Amplitude and Frequency


def compute_mechanical_frequency_amplitude_fft(
        times: np.ndarray,
        signals: np.ndarray):

    dt_sig = times[1] - times[0]
    n_step = len(times)

    if signals.ndim == 1:
        signals = signals[:, None]   # (time, 1)
    elif signals.ndim != 2:
        raise ValueError(f"signals must be 1D or 2D, got {signals.shape}")

    # FFT along time
    fft_vals = np.fft.rfft(signals, axis=0)              # (n_freq, channels)
    freqs = np.fft.rfftfreq(n_step, d=dt_sig)     # (n_freq,)
    mag = np.abs(fft_vals)                         # (n_freq, channels)

    # ignore DC when searching peak
    peak_idx = np.argmax(mag[1:, :], axis=0) + 1   # (channels)
    peak_freq = freqs[peak_idx]                    # (channels,)

    # amplitude ≈ 2*|FFT|/N for non-DC bins
    peak_amp = 2.0 * mag[peak_idx,
                         np.arange(mag.shape[1])] / n_step  # (channels,)

    # If originally 1D, return scalars
    if signals.ndim == 1:
        return peak_freq[0], peak_amp[0]

    return peak_freq,  peak_amp

# MM2
# Mechanical Speed


def get_robot_direction_pca(
    coordinates_xy: np.ndarray,
    n_links_pca: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute the PCA of the links positions at a given step
    Returns the forward and left direction based on the PCA and the tail-head axis.
    '''

    cov_mat = np.cov(
        [
            coordinates_xy[step, :n_links_pca, 0],
            coordinates_xy[step, :n_links_pca, 1],
        ]
    )
    eig_values, eig_vecs = np.linalg.eig(cov_mat)
    largest_index = np.argmax(eig_values)
    direction_fwd = eig_vecs[:, largest_index]

    # Align the direction with the tail-head axis
    p_tail2head = coordinates_xy[step, 0] - coordinates_xy[step, n_links_pca-1]
    direction_sign = np.sign(np.dot(p_tail2head, direction_fwd))
    direction_fwd = direction_sign * direction_fwd

    direction_left = np.cross(
        [0, 0, 1],
        [direction_fwd[0], direction_fwd[1], 0]
    )[:2]

    return direction_fwd, direction_left


def compute_mechanical_speed(links_positions: np.ndarray,
                             links_velocities: np.ndarray):
    '''
    Computes the axial and lateral speed based on the PCA of the links positions
    '''

    n_steps = links_positions.shape[0]
    n_links = 9  # actuated links of Polymander along the spine
    links_pos_xy = links_positions[:, :, :2]
    links_vel_xy = links_velocities[:, :, :2]

    speed_forward = np.zeros(n_steps)
    speed_lateral = np.zeros(n_steps)

    n_links_total = links_positions.shape[1]
    masses = LINKS_MASSES[:n_links_total]
    mass_sum = float(np.sum(masses))

    for idx in range(n_steps):

        # Compute the PCA of the links positions
        direction_fwd, direction_left = get_robot_direction_pca(
            coordinates_xy=links_pos_xy,
            n_links_pca=n_links,
            step=idx,
        )

        # Center-of-mass velocity from link velocities
        v_com = np.sum(links_vel_xy[idx] * masses[:, None], axis=0) / mass_sum

        # Projections on PCA direction
        speed_forward[idx] = float(np.dot(v_com, direction_fwd))
        speed_lateral[idx] = float(np.dot(v_com, direction_left))

    return np.mean(speed_forward), np.mean(speed_lateral)

# MM3
# Curvature


def compute_trajectory_curvature(
    trajectory: np.ndarray,
    timestep: float,
    sim_fraction: float = 0.2,
    min_speed: float = 5e-2,
    use_abs: bool = False,
):
    ''' Compute the curvature of the trajectory '''

    dt     = timestep
    n_step = trajectory.shape[0]
    n_cons = round(n_step * sim_fraction)
    traj = trajectory[-n_cons:]

    # Compute gradients on lightly filtered COM to reduce numerical noise.
    x = medfilt(traj[:, 0], kernel_size=5)
    y = medfilt(traj[:, 1], kernel_size=5)

    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)

    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)

    # Compute speed
    speed_sq = dx**2 + dy**2
    speed = np.sqrt(speed_sq)

    # Ignore very-low-speed samples to avoid curvature blow-up at start/stop.
    adaptive_min_speed = max(min_speed, float(np.nanpercentile(speed, 20)))
    mask = speed > adaptive_min_speed

    # Compute curvature
    curvature       = np.full_like(speed_sq, np.nan)
    numerator       = dx * ddy - ddx * dy
    denominator     = speed_sq ** 1.5
    curvature[mask] = numerator[mask] / denominator[mask]

    # Median filter to remove outliers
    k_size         = 5
    curvature      = medfilt(curvature, kernel_size=5)
    valid_curvature = curvature[k_size:-k_size]
    if use_abs:
        valid_curvature = np.abs(valid_curvature)
    curvature_mean = np.nanmean(valid_curvature)

    return curvature_mean


# MM4
# Mechanical Engery and CoT


def compute_mechanical_energy_and_cot(times: np.ndarray,
                                      links_positions: np.ndarray,
                                      joints_torques: np.ndarray,
                                      joints_velocities: np.ndarray,
                                      ):
    """
    Compute sum of energy consumptions and CoT.
    Hint:
    Only take POSITIVE values during energy consumption (no energy storing of the active part)
    Compute the integration of traveled distance for the CoM of the robot (useful varibles: LINKS_MASSES, TOTAL_MASS)
    """

    dt = float(times[1] - times[0])

    # Power per joint: max(tau * theta_dot, 0)
    power = joints_torques * joints_velocities
    power_pos = np.maximum(power, 0.0)
    energy = float(dt * np.sum(power_pos))

    # CoM forward distance (use x as world-forward axis in the simulator)
    n_links_total = links_positions.shape[1]
    masses = LINKS_MASSES[:n_links_total]
    mass_sum = float(np.sum(masses))
    com_pos = np.sum(links_positions * masses[None, :, None], axis=1) / mass_sum
    d_fwd = float(com_pos[-1, 0] - com_pos[0, 0])
    d_fwd = max(abs(d_fwd), 1e-9)
    cot = float(energy / d_fwd)
    return energy, cot

