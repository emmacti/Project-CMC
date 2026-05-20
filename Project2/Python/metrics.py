"""Locomotion metrics for Project 2 (speed, CoT)."""

import numpy as np
from scipy.signal import medfilt

LINKS_MASSES = np.array(
    [
        0.328768, 0.274101, 0.107688, 0.107688, 0.107688,
        0.0433459, 0.107688, 0.107688,
        0.18959,
        0.0194482, 0.164364, 0.0194482, 0.164364,
        0.0194482, 0.164364, 0.0194482, 0.164364,
        0.321614, 0.164651,
    ]
)


def get_robot_direction_pca(coordinates_xy, n_links_pca, step):
    """PCA-based forward direction from body link positions."""
    cov_mat = np.cov(
        [
            coordinates_xy[step, :n_links_pca, 0],
            coordinates_xy[step, :n_links_pca, 1],
        ]
    )
    eig_values, eig_vecs = np.linalg.eig(cov_mat)
    largest_index = np.argmax(eig_values)
    direction_fwd = eig_vecs[:, largest_index]
    p_tail2head = coordinates_xy[step, 0] - coordinates_xy[step, n_links_pca - 1]
    direction_sign = np.sign(np.dot(p_tail2head, direction_fwd))
    direction_fwd = direction_sign * direction_fwd
    direction_left = np.cross(
        [0, 0, 1],
        [direction_fwd[0], direction_fwd[1], 0],
    )[:2]
    return direction_fwd, direction_left


def compute_mechanical_speed(links_positions, links_velocities):
    """Mean forward and lateral COM speed (PCA projection)."""
    n_steps = links_positions.shape[0]
    n_links = 9
    links_pos_xy = links_positions[:, :, :2]
    links_vel_xy = links_velocities[:, :, :2]
    speed_forward = np.zeros(n_steps)
    speed_lateral = np.zeros(n_steps)
    n_links_total = links_positions.shape[1]
    masses = LINKS_MASSES[:n_links_total]
    mass_sum = float(np.sum(masses))

    for idx in range(n_steps):
        direction_fwd, direction_left = get_robot_direction_pca(
            links_pos_xy, n_links, idx,
        )
        v_com = np.sum(links_vel_xy[idx] * masses[:, None], axis=0) / mass_sum
        speed_forward[idx] = float(np.dot(v_com, direction_fwd))
        speed_lateral[idx] = float(np.dot(v_com, direction_left))

    return float(np.mean(speed_forward)), float(np.mean(speed_lateral))


def compute_mechanical_energy_and_cot(
        times,
        links_positions,
        joints_torques,
        joints_velocities,
):
    """Mechanical energy (positive work) and cost of transport."""
    dt = float(times[1] - times[0])
    power = joints_torques * joints_velocities
    power_pos = np.maximum(power, 0.0)
    energy = float(dt * np.sum(power_pos))

    n_links_total = links_positions.shape[1]
    masses = LINKS_MASSES[:n_links_total]
    mass_sum = float(np.sum(masses))
    com_pos = np.sum(links_positions * masses[None, :, None], axis=1) / mass_sum
    d_fwd = float(com_pos[-1, 0] - com_pos[0, 0])
    d_fwd = max(abs(d_fwd), 1e-9)
    cot = float(energy / d_fwd)
    return energy, cot
