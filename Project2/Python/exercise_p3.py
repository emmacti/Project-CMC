"""Exercise 3: Limb and Spine Coordination while walking"""

import os
os.environ.setdefault('MUJOCO_GL', 'glfw')
import numpy as np
from salamandra_simulation.simulation import simulation, simulation_sweep
from simulation_parameters import SimulationParameters
from farms_core import pylog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_results import (
    load_data,
    compute_speed,
)

RESULTS_DIR = './results/'
os.makedirs(RESULTS_DIR, exist_ok=True)


def _links_velocity_from_positions(links_positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute link velocities from positions with finite differences."""
    if links_positions.shape[0] < 2:
        return np.zeros_like(links_positions)
    vel = np.zeros_like(links_positions, dtype=float)
    vel[1:] = np.diff(links_positions, axis=0) / float(dt)
    vel[0] = vel[1]
    return vel


def _compute_cot(exp_data, nsteps_considered=400) -> float:
    """Compute Cost of Transport: E_positive / D_forward.

    Only positive mechanical power is counted (consistent with Project 1):
      P_j(t) = max(τ_j · q̇_j, 0)
    Negative power (passive braking) is excluded because we are interested
    in the metabolic-like cost of active muscle effort.
    The floor on distance (1e-6 m) prevents division-by-zero for stationary
    robots, but the resulting CoT is meaningless — callers should filter with
    _MIN_WALK_SPEED before interpreting CoT values.
    """
    data = exp_data.animats[0]
    dt = float(exp_data.timestep)
    joints_vel = np.asarray(data.sensors.joints.velocities_all())
    joints_tau = np.asarray(data.sensors.joints.motor_torques_all())
    power = np.maximum(joints_tau * joints_vel, 0.0)
    energy = float(np.sum(power[-nsteps_considered:]) * dt)
    links_positions = np.asarray(data.sensors.links.urdf_positions())
    links_vel = (
        np.asarray(data.sensors.links.urdf_velocities())
        if hasattr(data.sensors.links, 'urdf_velocities')
        else _links_velocity_from_positions(links_positions, dt=dt)
    )
    speed_forward, _ = compute_speed(
        links_positions=links_positions,
        links_vel=links_vel,
        nsteps_considered=nsteps_considered,
    )
    distance = max(1e-6, float(speed_forward * (nsteps_considered * dt)))
    return float(energy / distance)


def _get_steady_state(data, dt, t_steady=5.0):
    """Slice the last t_steady seconds of a simulation for steady-state analysis.

    Returns (times_ss, osc_phases_ss, joint_angles_ss, links_positions_full, t_start_idx).
    Note: links_positions is returned unsliced so callers can plot the full
    trajectory while still using only the steady-state window for metrics.
    """
    osc_phases = np.asarray(data.state.phases_all())
    joint_angles = np.asarray(data.sensors.joints.positions_all())
    links_positions = np.asarray(data.sensors.links.urdf_positions())
    T = osc_phases.shape[0]
    times = np.arange(T) * dt
    t_start = max(0, T - int(t_steady / dt))
    return times[t_start:], osc_phases[t_start:], joint_angles[t_start:], links_positions, t_start


# ── Exercise 3.1 – Nominal walking analysis ──────────────────────────────────

def exercise_3_1_walking_analysis(timestep):
    """Exercise 3.1 – Spine and limb phases during nominal walking (drive=2.5).

    Simulation data saved to logs/ex3_1/.
    Plots saved to results/ as ex3_1_*.png.
    """
    log_dir = './logs/ex3_1/'
    os.makedirs(log_dir, exist_ok=True)

    # drive=2.5 is in the middle of the walking regime (1.0 < drive < 3.0)
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        drive=2.5,
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        headless=True,
        output=log_dir + 'sim_0',
        verbose=False,
    )

    exp_data, _ = load_data(log_dir + 'sim_{}', 0)
    data = exp_data.animats[0]
    dt = float(exp_data.timestep)
    times_ss, osc_phases, joint_angles, links_positions, t_start = _get_steady_state(data, dt)

    # Oscillator layout: 16 body oscillators (2 per joint × 8 joints),
    # indices 2j = left chain, 2j+1 = right chain for body joint j.
    # Then 16 limb oscillators starting at index 16.
    n_body_joints = 8

    # ── 1. Spine oscillator phases ────────────────────────────────────────────
    # np.unwrap removes 2π jumps so the cumulative phase progression is visible.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j in range(n_body_joints):
        axes[0].plot(times_ss, np.unwrap(osc_phases[:, 2*j]),   label=f'J{j+1}')
        axes[1].plot(times_ss, np.unwrap(osc_phases[:, 2*j+1]), label=f'J{j+1}')
    axes[0].set_ylabel('Phase (rad) – left chain')
    axes[1].set_ylabel('Phase (rad) – right chain')
    axes[1].set_xlabel('Time (s)')
    axes[0].set_title('Spine oscillator phases – nominal walking (drive=2.5)')
    for ax in axes:
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3_1_spine_phases.png', dpi=150)
    plt.close()

    # Phase lag between adjacent joints (left chain)
    phase_lags = [
        float(np.mean(np.unwrap(osc_phases[:, 2*(j+1)]) - np.unwrap(osc_phases[:, 2*j])))
        for j in range(n_body_joints - 1)
    ]
    pylog.info('Ex3.1 phase lags per segment (rad): %s', np.round(phase_lags, 3))

    # ── 2. Body joint angles ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for j in range(n_body_joints):
        ax.plot(times_ss, joint_angles[:, j], label=f'J{j+1}', alpha=0.85)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint angle (rad)')
    ax.set_title('Body joint angles – traveling wave during walking')
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3_1_joint_angles.png', dpi=150)
    plt.close()

    # ── 3. Limb joint angles ──────────────────────────────────────────────────
    limb_labels = [
        'LF shldr', 'LF elbow', 'RF shldr', 'RF elbow',
        'LH hip',   'LH knee',  'RH hip',   'RH knee',
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes.flat, limb_labels)):
        ax.plot(times_ss, joint_angles[:, n_body_joints + k])
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Angle (rad)', fontsize=8)
        ax.grid(True)
    fig.suptitle('Limb joint angles – trot gait (drive=2.5)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3_1_limb_angles.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── 4. Head trajectory ────────────────────────────────────────────────────
    head_pos = links_positions[:, 0, :]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(head_pos[:, 0], head_pos[:, 1], 'b', linewidth=1.2)
    ax.plot(head_pos[0, 0],  head_pos[0, 1],  'go', label='Start')
    ax.plot(head_pos[-1, 0], head_pos[-1, 1], 'rs', label='End')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Head trajectory – nominal walking (drive=2.5)')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3_1_trajectory.png', dpi=150)
    plt.close()

    pylog.info('Ex3.1 plots saved to %s', RESULTS_DIR)


# ── Exercise 3.2 – Disabled spine-limb coupling ───────────────────────────────

def exercise_3_2_disable_coupling(timestep):
    """Exercise 3.2 – Walk with spine-limb coupling disabled.

    Video saved to logs/ex3_2_disable_coupling/.
    Analysis plots comparing coupled vs decoupled saved to results/ as ex3_2_*.png.
    """
    log_dir_coupled   = './logs/ex3_1/'          # reuse q3.1 data if available
    log_dir_decoupled = './logs/ex3_2_disable_coupling/'
    os.makedirs(log_dir_decoupled, exist_ok=True)

    # ── Run decoupled simulation (headless for data) ──────────────────────────
    # Why initial_phase_seed=0 (different from the default 42 used in 3.1):
    #   With the same seed, spine and limb oscillators start with an identical
    #   relative phase in both conditions. Since the intrinsic frequencies are
    #   also very similar after the fix, the uncoupled system maintains that
    #   initial offset indefinitely — making both conditions look identical.
    #   A different seed gives an arbitrary spine-limb initial phase offset.
    #   With coupling (3.1), that offset is corrected by the CPG; without it
    #   (here), it persists, revealing the actual effect of decoupling.
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        drive=2.5,
        disable_limb_spine_coupling=True,
        initial_phase_seed=0,
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        headless=True,
        output=log_dir_decoupled + 'sim_0',
        verbose=False,
    )

    # ── Record video separately (requires mjpython on macOS) ─────────────────
    try:
        simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=False,
            record=True,
            output=log_dir_decoupled + 'sim_0_video',
            record_path=log_dir_decoupled + 'ex3_2_disable_coupling.mp4',
            verbose=False,
        )
    except RuntimeError as e:
        pylog.warning('Video recording skipped (run with mjpython on macOS): %s', e)

    # ── Also run coupled simulation if not already done ───────────────────────
    coupled_hdf5 = log_dir_coupled + 'sim_0/simulation.hdf5'
    if not os.path.exists(coupled_hdf5):
        os.makedirs(log_dir_coupled, exist_ok=True)
        simulation(
            sim_parameters=SimulationParameters(
                duration=15, timestep=timestep, drive=2.5,
            ),
            arena='land',
            fast=True,
            headless=True,
            output=log_dir_coupled + 'sim_0',
            verbose=False,
        )

    # ── Load both datasets ────────────────────────────────────────────────────
    exp_coupled,   _ = load_data(log_dir_coupled   + 'sim_{}', 0)
    exp_decoupled, _ = load_data(log_dir_decoupled + 'sim_{}', 0)

    dt = float(exp_coupled.timestep)

    data_c = exp_coupled.animats[0]
    data_d = exp_decoupled.animats[0]

    times_c, phases_c, angles_c, links_c, _ = _get_steady_state(data_c, dt)
    times_d, phases_d, angles_d, links_d, _ = _get_steady_state(data_d, dt)

    n_body_joints = 8
    n_body_osc    = 16   # 2 oscillators × 8 body joints

    # ── Plot 1: spine vs limb phase difference ────────────────────────────────
    # We compare the phase of the body oscillator at the girdle attachment point
    # to the phase of the corresponding proximal limb oscillator:
    #   Forelimbs attach at body joint 1  → body osc index 2*1 = 2
    #   Hindlimbs attach at body joint 6  → body osc index 2*6 = 12
    #   LF shoulder = limb joint 0        → osc index 16 + 2*0 = 16
    #   LH hip      = limb joint 4        → osc index 16 + 2*4 = 24
    # In the coupled case the difference should be stable (locked by coupling).
    # In the decoupled case it may drift or sit at a different arbitrary offset.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for title, phases, ax in [
        ('Coupled', phases_c, axes[0]),
        ('Decoupled', phases_d, axes[1]),
    ]:
        fore_body = np.unwrap(phases[:, 2*1])            # body joint 1, left osc
        fore_limb = np.unwrap(phases[:, n_body_osc + 0]) # LF shoulder osc
        hind_body = np.unwrap(phases[:, 2*6])            # body joint 6, left osc
        hind_limb = np.unwrap(phases[:, n_body_osc + 8]) # LH hip osc
        times = times_c if title == 'Coupled' else times_d
        ax.plot(times, fore_body - fore_limb, label='Fore: body–limb Δφ')
        ax.plot(times, hind_body - hind_limb, label='Hind: body–limb Δφ')
        ax.set_title(f'Spine–limb phase difference ({title})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Δφ (rad)')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3_2_phase_difference.png', dpi=150)
    plt.close()

    # ── Plot 2: body joint angles comparison ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for title, times, angles, ax in [
        ('Coupled',   times_c, angles_c, axes[0]),
        ('Decoupled', times_d, angles_d, axes[1]),
    ]:
        for j in range(n_body_joints):
            ax.plot(times, angles[:, j], alpha=0.7, label=f'J{j+1}')
        ax.set_title(f'Body joint angles ({title})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3_2_body_angles.png', dpi=150)
    plt.close()

    # ── Plot 3: trajectory + speed comparison ────────────────────────────────
    links_vel_c = _links_velocity_from_positions(links_c, dt)
    links_vel_d = _links_velocity_from_positions(links_d, dt)
    speed_c, _ = compute_speed(links_c, links_vel_c)
    speed_d, _ = compute_speed(links_d, links_vel_d)
    cot_c = _compute_cot(exp_coupled)
    cot_d = _compute_cot(exp_decoupled)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Trajectories
    for title, lp, color in [('Coupled', links_c, 'steelblue'), ('Decoupled', links_d, 'tomato')]:
        head = lp[:, 0, :]
        axes[0].plot(head[:, 0], head[:, 1], color=color, label=title, linewidth=1.2)
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_title('Head trajectories')
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True)

    # Speed bar
    axes[1].bar(['Coupled', 'Decoupled'], [speed_c, speed_d],
                color=['steelblue', 'tomato'])
    axes[1].set_ylabel('Forward speed (m/s)')
    axes[1].set_title('Walking speed comparison')
    axes[1].grid(True, axis='y')
    for i, v in enumerate([speed_c, speed_d]):
        axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)

    # CoT bar
    axes[2].bar(['Coupled', 'Decoupled'], [cot_c, cot_d],
                color=['steelblue', 'tomato'])
    axes[2].set_ylabel('Cost of Transport')
    axes[2].set_title('Energy efficiency comparison')
    axes[2].grid(True, axis='y')
    for i, v in enumerate([cot_c, cot_d]):
        axes[2].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3_2_speed_cot_comparison.png', dpi=150)
    plt.close()

    pylog.info(
        'Ex3.2 – Coupled: speed=%.3f m/s, CoT=%.2f | Decoupled: speed=%.3f m/s, CoT=%.2f',
        speed_c, cot_c, speed_d, cot_d,
    )
    pylog.info('Ex3.2 plots saved to %s', RESULTS_DIR)


# ── Exercise 3.3 – Limb-spine anti-phase ─────────────────────────────────────

def _record_video(sim_parameters, log_dir, filename):
    """Run a simulation and attempt video recording.

    Two-pass strategy:
      1. Headless fast run  → saves simulation.hdf5 for data analysis.
      2. GUI slow run       → records the .mp4 video.
    The two passes are separated because MuJoCo's passive viewer (needed for
    recording) requires `mjpython` on macOS, while the headless run works with
    any Python interpreter. Keeping them separate ensures analysis data is
    always saved even when video recording is unavailable.
    To record on macOS run:  mjpython exercise_p3.py
    """
    os.makedirs(log_dir, exist_ok=True)
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        headless=True,
        output=log_dir + 'sim_0',
        verbose=False,
    )
    try:
        simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=False,
            record=True,
            output=log_dir + 'sim_0_video',
            record_path=log_dir + filename,
            verbose=False,
        )
        pylog.info('Video saved to %s%s', log_dir, filename)
    except RuntimeError as e:
        pylog.warning('Video skipped (run with mjpython on macOS): %s', e)


def exercise_3a_optimal_video(timestep):
    """Video of optimal walking (phi and drive that maximise speed from sweep)."""
    rs = np.load('./logs/ex3a/results_speed.npy')
    best = rs[np.argmax(rs[:, 2])]
    best_phi, best_drive = float(best[0]), float(best[1])
    pylog.info('Optimal: phi=%.3f rad (%.1f°), drive=%.3f', best_phi,
               np.degrees(best_phi), best_drive)
    _record_video(
        SimulationParameters(duration=15, timestep=timestep,
                             drive=best_drive,
                             limb_spine_phase_offset=best_phi),
        log_dir='./logs/ex3a_optimal/',
        filename='ex3a_optimal_walk.mp4',
    )


def exercise_3a_antiphase_video(timestep):
    """Video of anti-phase walking: phi = optimal + π (worst coordination).

    If the optimal phase offset is phi_opt, the 'anti-phase to ideal' is the
    offset exactly half a cycle away: phi_opt + π.  For our sweep result where
    phi_opt ≈ -π ≡ +π, this gives phi_anti = 0 (in-phase coupling), which
    corresponds to the lowest forward speed observed in the sweep.
    """
    rs = np.load('./logs/ex3a/results_speed.npy')
    best_phi = float(rs[np.argmax(rs[:, 2]), 0])
    best_drive = float(rs[np.argmax(rs[:, 2]), 1])
    # Shift by π then wrap back to [-π, π] to stay within the parameter range.
    anti_phi = ((best_phi + np.pi) + np.pi) % (2 * np.pi) - np.pi
    pylog.info('Anti-phase phi=%.3f rad (%.1f°), drive=%.3f',
               anti_phi, np.degrees(anti_phi), best_drive)
    _record_video(
        SimulationParameters(duration=15, timestep=timestep,
                             drive=best_drive,
                             limb_spine_phase_offset=anti_phi),
        log_dir='./logs/ex3a_antiphase/',
        filename='ex3a_antiphase_walk.mp4',
    )


# ── Exercise 3a – Phase offset sweep ─────────────────────────────────────────

def exercise_3a_coordination(timestep):
    """Exercise 3a – 2D sweep: forward speed and CoT vs phase offset × drive.

    Skips the sweep if results already exist. Re-run by deleting logs/ex3a/.
    Saves plots to results/ as ex3a_*.png.
    """
    log_dir = './logs/ex3a/'
    os.makedirs(log_dir, exist_ok=True)

    # Walking regime: drive ∈ [1.0, 3.0] based on CPG frequency/amplitude
    # formulas in robot_parameters.py. We stay just inside both ends to avoid
    # edge effects near the swim transition (~3.0) and the silence threshold (1.0).
    drives        = np.linspace(1.5, 3.2, 5)
    # Full circle sampled at 9 evenly-spaced points including ±π.
    phase_offsets = np.linspace(-np.pi, np.pi, 9)

    # ── Run sweep only if needed ──────────────────────────────────────────────
    speed_file = log_dir + 'results_speed.npy'
    cot_file   = log_dir + 'results_cot.npy'
    if not (os.path.exists(speed_file) and os.path.exists(cot_file)):
        parameter_set = [
            SimulationParameters(
                duration=10,
                timestep=timestep,
                drive=float(d),
                limb_spine_phase_offset=float(phi),
            )
            for d in drives
            for phi in phase_offsets
        ]
        sim_args = [
            {
                'sim_parameters': p,
                'arena': 'land',
                'fast': True,
                'headless': True,
                'output': f'{log_dir}sim_{i}',
                'verbose': False,
            }
            for i, p in enumerate(parameter_set)
        ]
        simulation_sweep(sim_args, processes=4)

        results_speed, results_cot, results_lat = [], [], []
        for i, p in enumerate(parameter_set):
            exp_data, _ = load_data(log_dir + 'sim_{}', i)
            data = exp_data.animats[0]
            dt = float(exp_data.timestep)
            lp = np.asarray(data.sensors.links.urdf_positions())
            lv = (
                np.asarray(data.sensors.links.urdf_velocities())
                if hasattr(data.sensors.links, 'urdf_velocities')
                else _links_velocity_from_positions(lp, dt)
            )
            speed_fwd, speed_lat = compute_speed(lp, lv)
            cot = _compute_cot(exp_data)
            # Lateral deviation: std of head y-position in last 5 s
            n_ss = min(1000, lp.shape[0])
            head_y_std = float(np.std(lp[-n_ss:, 0, 1]))
            results_speed.append([p.limb_spine_phase_offset, p.drive, speed_fwd])
            results_cot.append([p.limb_spine_phase_offset, p.drive, cot])
            results_lat.append([p.limb_spine_phase_offset, p.drive, head_y_std])

        results_speed = np.asarray(results_speed, dtype=float)
        results_cot   = np.asarray(results_cot,   dtype=float)
        results_lat   = np.asarray(results_lat,   dtype=float)
        np.save(speed_file, results_speed)
        np.save(cot_file,   results_cot)
        np.save(log_dir + 'results_lat.npy', results_lat)
    else:
        pylog.info('Ex3a: loading existing sweep results (delete logs/ex3a/ to re-run)')
        results_speed = np.load(speed_file)
        results_cot   = np.load(cot_file)
        lat_file = log_dir + 'results_lat.npy'
        if os.path.exists(lat_file):
            results_lat = np.load(lat_file)
        else:
            results_lat = None

    # ── Find optima ───────────────────────────────────────────────────────────
    # No speed masking here: CoT is meaningful for all walking drives in 3a
    # because the drive always produces forward motion (unlike 3b gains=0).
    best_speed_idx  = int(np.argmax(results_speed[:, 2]))
    best_cot_idx    = int(np.argmin(results_cot[:, 2]))
    best_phi_speed  = results_speed[best_speed_idx, 0]
    best_drive_speed = results_speed[best_speed_idx, 1]
    best_phi_cot    = results_cot[best_cot_idx, 0]
    best_drive_cot  = results_cot[best_cot_idx, 1]
    anti_phi = ((best_phi_speed + np.pi) + np.pi) % (2 * np.pi) - np.pi

    pylog.info('Ex3a best speed : phi=%.3f rad (%.1f°), drive=%.3f -> %.4f m/s',
               best_phi_speed, np.degrees(best_phi_speed), best_drive_speed,
               results_speed[best_speed_idx, 2])
    pylog.info('Ex3a best CoT   : phi=%.3f rad (%.1f°), drive=%.3f -> CoT=%.4f',
               best_phi_cot, np.degrees(best_phi_cot), best_drive_cot,
               results_cot[best_cot_idx, 2])
    pylog.info('Ex3a anti-phase : phi=%.3f rad (%.1f°)',
               anti_phi, np.degrees(anti_phi))

    from plot_results import plot_2d

    # ── Plot 1: 2D heatmap – forward speed ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_2d(results_speed, labels=['Phase offset (rad)', 'Drive', 'Forward speed (m/s)'])
    ax.plot(best_phi_speed, best_drive_speed, 'w*', ms=14,
            label=f'Best speed: φ={best_phi_speed:.2f} rad')
    ax.plot(anti_phi, best_drive_speed, 'rx', ms=10, mew=2,
            label=f'Anti-phase: φ={anti_phi:.2f} rad')
    ax.legend(fontsize=8)
    ax.set_title('Forward speed vs phase offset and drive')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3a_speed_heatmap.png', dpi=150)
    plt.close()

    # ── Plot 2: 2D heatmap – CoT ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_2d(results_cot, labels=['Phase offset (rad)', 'Drive', 'Cost of Transport'])
    ax.plot(best_phi_cot, best_drive_cot, 'w*', ms=14,
            label=f'Min CoT: φ={best_phi_cot:.2f} rad')
    ax.legend(fontsize=8)
    ax.set_title('Cost of Transport vs phase offset and drive')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3a_cot_heatmap.png', dpi=150)
    plt.close()

    # ── Plot 3: 1D slices at best drive ──────────────────────────────────────
    # atol=0.05 tolerates floating-point rounding in the stored drive values.
    best_drive_mask = np.isclose(results_speed[:, 1], best_drive_speed, atol=0.05)
    phis_slice  = results_speed[best_drive_mask, 0]
    speed_slice = results_speed[best_drive_mask, 2]
    cot_slice   = results_cot[best_drive_mask, 2]
    sort_idx    = np.argsort(phis_slice)
    phis_slice  = phis_slice[sort_idx]
    speed_slice = speed_slice[sort_idx]
    cot_slice   = cot_slice[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(phis_slice, speed_slice, 'o-', color='steelblue', linewidth=1.8)
    axes[0].axvline(best_phi_speed, color='gold',  ls='--', label=f'Best speed φ={best_phi_speed:.2f}')
    axes[0].axvline(anti_phi,       color='tomato', ls='--', label=f'Anti-phase φ={anti_phi:.2f}')
    axes[0].set_xlabel('Phase offset φ (rad)')
    axes[0].set_ylabel('Forward speed (m/s)')
    axes[0].set_title(f'Speed vs φ (drive={best_drive_speed:.2f})')
    axes[0].legend(fontsize=8); axes[0].grid(True)
    # π-tick formatting
    axes[0].set_xticks(np.linspace(-np.pi, np.pi, 5))
    axes[0].set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    axes[1].plot(phis_slice, cot_slice, 'o-', color='darkorange', linewidth=1.8)
    axes[1].axvline(best_phi_cot, color='gold',  ls='--', label=f'Min CoT φ={best_phi_cot:.2f}')
    axes[1].axvline(anti_phi,     color='tomato', ls='--', label=f'Anti-phase φ={anti_phi:.2f}')
    axes[1].set_xlabel('Phase offset φ (rad)')
    axes[1].set_ylabel('Cost of Transport')
    axes[1].set_title(f'CoT vs φ (drive={best_drive_speed:.2f})')
    axes[1].legend(fontsize=8); axes[1].grid(True)
    axes[1].set_xticks(np.linspace(-np.pi, np.pi, 5))
    axes[1].set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3a_1d_slices.png', dpi=150)
    plt.close()

    # ── Plot 4: lateral deviation heatmap (if available) ─────────────────────
    if results_lat is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_2d(results_lat, labels=['Phase offset (rad)', 'Drive', 'Head lateral std (m)'])
        ax.plot(best_phi_speed, best_drive_speed, 'w*', ms=14,
                label=f'Best speed: φ={best_phi_speed:.2f}')
        ax.legend(fontsize=8)
        ax.set_title('Walking straightness (head lateral std) vs phase offset and drive')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR + 'ex3a_straightness_heatmap.png', dpi=150)
        plt.close()

    pylog.info('Ex3a plots saved to %s', RESULTS_DIR)


# ── Exercise 3b – Amplitude gain sweep ───────────────────────────────────────

# Minimum forward speed to consider a simulation "walking" for CoT purposes.
# Below this the robot is barely moving; its CoT is dominated by near-zero
# distance and is physically meaningless as a locomotion efficiency metric.
_MIN_WALK_SPEED = 0.10  # m/s


def exercise_3b_coordination(timestep):
    """Exercise 3b – 2D sweep: speed and CoT vs axial × limb amplitude gains.

    Uses the optimal phase offset and drive from Ex3a.
    Skips the sweep if results already exist (delete logs/ex3b/ to re-run).
    Saves plots to results/ as ex3b_*.png.
    """
    log_dir = './logs/ex3b/'
    os.makedirs(log_dir, exist_ok=True)

    # ── Pull optimal settings from Ex3a ──────────────────────────────────────
    best_phi = 0.0
    best_drive = 2.5
    try:
        rs_a = np.load('./logs/ex3a/results_speed.npy')
        best_idx   = int(np.argmax(rs_a[:, 2]))
        best_phi   = float(rs_a[best_idx, 0])
        best_drive = float(rs_a[best_idx, 1])
        pylog.info('Ex3b: using Ex3a optimal phi=%.3f rad, drive=%.3f',
                   best_phi, best_drive)
    except Exception:
        pylog.warning('Ex3a results not found; using phi=0, drive=2.5')

    # Gain range 0→2: starts from 0 (no movement) and extends to 2× the default
    # amplitude, which is enough to clearly show performance deterioration at
    # high limb gains while covering the interesting transition region near 1.0.
    axial_gains = np.linspace(0.0, 2.0, 9)
    limb_gains  = np.linspace(0.0, 2.0, 9)

    # ── Run sweep only if needed ──────────────────────────────────────────────
    speed_file = log_dir + 'results_speed.npy'
    cot_file   = log_dir + 'results_cot.npy'
    if not (os.path.exists(speed_file) and os.path.exists(cot_file)):
        parameter_set = [
            SimulationParameters(
                duration=10,
                timestep=timestep,
                drive=best_drive,
                limb_spine_phase_offset=best_phi,
                axial_amp_gain=float(ga),
                limb_amp_gain=float(gl),
            )
            for ga in axial_gains
            for gl in limb_gains
        ]
        sim_args = [
            {
                'sim_parameters': p,
                'arena': 'land',
                'fast': True,
                'headless': True,
                'output': f'{log_dir}sim_{i}',
                'verbose': False,
            }
            for i, p in enumerate(parameter_set)
        ]
        simulation_sweep(sim_args, processes=4)

        results_speed, results_cot = [], []
        for i, p in enumerate(parameter_set):
            exp_data, _ = load_data(log_dir + 'sim_{}', i)
            data = exp_data.animats[0]
            dt = float(exp_data.timestep)
            lp = np.asarray(data.sensors.links.urdf_positions())
            lv = (
                np.asarray(data.sensors.links.urdf_velocities())
                if hasattr(data.sensors.links, 'urdf_velocities')
                else _links_velocity_from_positions(lp, dt)
            )
            speed_fwd, _ = compute_speed(lp, lv)
            cot = _compute_cot(exp_data)
            results_speed.append([p.axial_amp_gain, p.limb_amp_gain, speed_fwd])
            results_cot.append([p.axial_amp_gain, p.limb_amp_gain, cot])

        results_speed = np.asarray(results_speed, dtype=float)
        results_cot   = np.asarray(results_cot,   dtype=float)
        np.save(speed_file, results_speed)
        np.save(cot_file,   results_cot)
    else:
        pylog.info('Ex3b: loading existing sweep (delete logs/ex3b/ to re-run)')
        results_speed = np.load(speed_file)
        results_cot   = np.load(cot_file)

    # ── Sanitise results ──────────────────────────────────────────────────────
    # CoT = E / distance explodes when the robot barely moves (e.g. gain = 0).
    # We set CoT to NaN for non-walking states so the heatmap interpolation
    # ignores them rather than pulling the colour scale toward huge values.
    walking_mask = results_speed[:, 2] >= _MIN_WALK_SPEED
    results_cot_clean = results_cot.copy()
    results_cot_clean[~walking_mask, 2] = np.nan

    # Negative speeds (robot moved backward) are clipped to 0 for display.
    # This occurs at extreme gains (e.g. axial=2, limb=2) where overexcitation
    # destabilises the gait; it is a real physical result, not an artefact.
    results_speed_clean = results_speed.copy()
    results_speed_clean[:, 2] = np.clip(results_speed_clean[:, 2], 0.0, None)

    # ── Find optima (walking states only) ─────────────────────────────────────
    walk_speed = results_speed_clean[walking_mask, 2]
    walk_rows  = results_speed_clean[walking_mask]
    best_speed_row = walk_rows[np.argmax(walk_speed)]
    best_axial_speed = best_speed_row[0]
    best_limb_speed  = best_speed_row[1]

    cot_valid = results_cot_clean[walking_mask, 2]
    cot_rows  = results_cot_clean[walking_mask]
    best_cot_row   = cot_rows[np.nanargmin(cot_valid)]
    best_axial_cot = best_cot_row[0]
    best_limb_cot  = best_cot_row[1]

    pylog.info('Ex3b best speed (walking): axial=%.2f, limb=%.2f -> %.4f m/s',
               best_axial_speed, best_limb_speed, best_speed_row[2])
    pylog.info('Ex3b best CoT  (walking): axial=%.2f, limb=%.2f -> CoT=%.4f',
               best_axial_cot, best_limb_cot, best_cot_row[2])
    pylog.info('Ex3b default (1,1) speed: %.4f m/s',
               float(results_speed_clean[
                   (np.isclose(results_speed_clean[:,0], 1.0, atol=0.1)) &
                   (np.isclose(results_speed_clean[:,1], 1.0, atol=0.1)), 2][0]))

    from plot_results import plot_2d

    # ── Plot 1: 2D heatmap – forward speed ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_2d(results_speed_clean,
            labels=['Axial amplitude gain', 'Limb amplitude gain',
                    'Forward speed (m/s)'])
    ax.plot(best_axial_speed, best_limb_speed, 'w*', ms=14,
            label=f'Best speed\n(a={best_axial_speed:.2f}, l={best_limb_speed:.2f})')
    ax.plot(1.0, 1.0, 'ws', ms=10, mew=1.5,
            label='Default (1.0, 1.0)')
    ax.legend(fontsize=8)
    ax.set_title('Forward speed vs amplitude gains (drive=%.2f, φ=%.2f rad)'
                 % (best_drive, best_phi))
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3b_speed_heatmap.png', dpi=150)
    plt.close()

    # ── Plot 2: 2D heatmap – CoT (walking states only) ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_2d(results_cot_clean,
            labels=['Axial amplitude gain', 'Limb amplitude gain',
                    'Cost of Transport'])
    ax.plot(best_axial_cot, best_limb_cot, 'w*', ms=14,
            label=f'Min CoT\n(a={best_axial_cot:.2f}, l={best_limb_cot:.2f})')
    ax.plot(1.0, 1.0, 'ws', ms=10, mew=1.5, label='Default (1.0, 1.0)')
    ax.legend(fontsize=8)
    ax.set_title('Cost of Transport vs amplitude gains (walking states only)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3b_cot_heatmap.png', dpi=150)
    plt.close()

    # ── Plot 3: 1D slices – vary one gain while the other is at default (1.0) ──
    # atol=0.13 is half the step size between gains (linspace step = 0.25),
    # so it selects exactly the row/column at gain=1.0 without ambiguity.
    limb_default_mask  = np.isclose(results_speed_clean[:, 1], 1.0, atol=0.13)
    axial_default_mask = np.isclose(results_speed_clean[:, 0], 1.0, atol=0.13)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, (mask, xlabel, xdata_col) in enumerate([
        (limb_default_mask,  'Axial amplitude gain', 0),
        (axial_default_mask, 'Limb amplitude gain',  1),
    ]):
        x  = results_speed_clean[mask, xdata_col]
        sp = results_speed_clean[mask, 2]
        ct = results_cot_clean[mask, 2]
        sort_i = np.argsort(x)
        x, sp, ct = x[sort_i], sp[sort_i], ct[sort_i]

        axes[0, col].plot(x, sp, 'o-', color='steelblue', linewidth=1.8)
        axes[0, col].axvline(1.0, color='k', ls='--', label='Default gain=1.0')
        axes[0, col].set_xlabel(xlabel)
        axes[0, col].set_ylabel('Forward speed (m/s)')
        fixed_lbl = 'limb=1.0' if col == 0 else 'axial=1.0'
        axes[0, col].set_title(f'Speed vs {xlabel.split()[0].lower()} gain ({fixed_lbl})')
        axes[0, col].legend(fontsize=8); axes[0, col].grid(True)

        axes[1, col].plot(x, ct, 'o-', color='darkorange', linewidth=1.8)
        axes[1, col].axvline(1.0, color='k', ls='--', label='Default gain=1.0')
        axes[1, col].set_xlabel(xlabel)
        axes[1, col].set_ylabel('Cost of Transport')
        axes[1, col].set_title(f'CoT vs {xlabel.split()[0].lower()} gain ({fixed_lbl})')
        axes[1, col].legend(fontsize=8); axes[1, col].grid(True)

    plt.suptitle('1D slices: effect of each amplitude gain with the other fixed at 1.0')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'ex3b_1d_slices.png', dpi=150)
    plt.close()

    pylog.info('Ex3b plots saved to %s', RESULTS_DIR)


def exercise_3b_optimal_video(timestep):
    """Video of optimal amplitude walking (gains that maximise speed)."""
    speed_file = './logs/ex3b/results_speed.npy'
    if not os.path.exists(speed_file):
        pylog.warning('Run exercise_3b_coordination first.')
        return

    rs = np.load(speed_file)
    # Load optimal phi/drive from ex3a
    best_phi, best_drive = 0.0, 2.5
    try:
        rs_a = np.load('./logs/ex3a/results_speed.npy')
        best_idx   = int(np.argmax(rs_a[:, 2]))
        best_phi   = float(rs_a[best_idx, 0])
        best_drive = float(rs_a[best_idx, 1])
    except Exception:
        pass

    # Use only walking-state optima (speed >= threshold)
    walking_mask = rs[:, 2] >= _MIN_WALK_SPEED
    rs_walk = rs[walking_mask]
    best = rs_walk[np.argmax(rs_walk[:, 2])]
    best_axial, best_limb = float(best[0]), float(best[1])
    pylog.info('Ex3b optimal video: axial=%.2f, limb=%.2f, drive=%.3f, phi=%.3f',
               best_axial, best_limb, best_drive, best_phi)

    _record_video(
        SimulationParameters(
            duration=15, timestep=timestep,
            drive=best_drive,
            limb_spine_phase_offset=best_phi,
            axial_amp_gain=best_axial,
            limb_amp_gain=best_limb,
        ),
        log_dir='./logs/ex3b_optimal/',
        filename='ex3b_optimal_walk.mp4',
    )


if __name__ == '__main__':
    exercise_3_1_walking_analysis(timestep=5e-3)
    exercise_3_2_disable_coupling(timestep=5e-3)
    exercise_3a_coordination(timestep=5e-3)      # sweep + rich plots
    exercise_3a_optimal_video(timestep=5e-3)     # video at best phi
    exercise_3a_antiphase_video(timestep=5e-3)   # video at best phi + π
    exercise_3b_coordination(timestep=5e-3)      # sweep + rich plots
    exercise_3b_optimal_video(timestep=5e-3)     # video at best amplitude gains
