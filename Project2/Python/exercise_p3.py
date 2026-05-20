"""Exercise 3: Limb and Spine Coordination while walking (Project 2 Part B)."""

import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from salamandra_simulation.simulation import simulation, simulation_sweep
from simulation_parameters import SimulationParameters
from plot_results import metrics_from_log_dir, plot_sweep_heatmaps
from farms_amphibious.data.data import AmphibiousExperimentData
import farms_pylog as pylog


# Default walking regime (Part A validation)
WALK_DRIVE = 2.0
WALK_DURATION = 15.0
SWEEP_DURATION = 12.0
OPTIMAL_PHASE_FILE = 'logs/ex3/optimal_phase_offset.pickle'


def _walk_params(timestep, duration=WALK_DURATION, drive=WALK_DRIVE, **kwargs):
    """Common SimulationParameters for land walking."""
    base = dict(
        duration=duration,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi / 2],
        drive=drive,
        phase_lag_body=2 * np.pi / 8,
        amplitude_gradient=None,
    )
    base.update(kwargs)
    return SimulationParameters(**base)


def _run_walk_sim(
        log_dir,
        sim_parameters,
        timestep,
        record=False,
        record_name=None,
        headless=True,
        fast=True,
):
    os.makedirs(log_dir, exist_ok=True)
    record_path = None
    if record and record_name:
        record_path = os.path.join(log_dir, record_name)
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=fast,
        headless=headless,
        output=log_dir,
        record=record,
        record_path=record_path,
        verbose=False,
    )


def exercise_3_spine_analysis(timestep):
    """Exercise 3.1: baseline walk + spine phase-lag analysis plots."""
    log_dir = 'logs/ex3_spine_analysis'
    sim_parameters = _walk_params(timestep)
    _run_walk_sim(
        log_dir, sim_parameters, timestep,
        record=False, headless=True, fast=True,
    )

    exp_data = AmphibiousExperimentData.from_file(
        os.path.join(log_dir, 'simulation.hdf5'),
    )
    data = exp_data.animats[0]
    # farms_core arrays do not support arithmetic on slices — convert to NumPy
    phases = np.asarray(data.state.array[:, :32], dtype=float)
    n_seg = 8
    # Left body chain oscillators: 0, 2, 4, ..., 14
    left_phases = phases[:, 0:16:2]
    # Steady-state window (last 4 s)
    n_steady = int(4.0 / timestep)
    left_steady = left_phases[-n_steady:]

    seg_lags = []
    for i in range(n_seg - 1):
        dphi = np.angle(
            np.exp(1j * (left_steady[:, i + 1] - left_steady[:, i])),
        )
        seg_lags.append(np.rad2deg(np.mean(dphi)))
    seg_lags = np.array(seg_lags)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), num='Ex3.1 spine analysis')
    t = np.arange(phases.shape[0]) * timestep
    for i in range(n_seg):
        axes[0].plot(t, left_phases[:, i], alpha=0.8, label=f'seg {i}')
    axes[0].set_ylabel('Phase [rad]')
    axes[0].set_title('Left body oscillator phases during walking')
    axes[0].legend(fontsize=6, ncol=4)
    axes[0].grid(True)

    axes[1].bar(range(len(seg_lags)), seg_lags)
    axes[1].axhline(
        np.rad2deg(2 * np.pi / n_seg), color='r', ls='--',
        label='nominal π/4 per segment',
    )
    axes[1].set_xlabel('Segment index (tail → head)')
    axes[1].set_ylabel('Mean phase lag [deg]')
    axes[1].set_title('Ipsilateral phase lags along spine (steady state)')
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    os.makedirs('results', exist_ok=True)
    fig.savefig('results/ex3_1_spine_phase_lags.png', dpi=200)
    pylog.info(
        'Spine segment lags [deg]: %s (mean %.1f)',
        np.round(seg_lags, 1), np.mean(seg_lags),
    )
    return seg_lags


def exercise_3_disable_limb_spine_coupling(timestep):
    """Exercise 3.2: walk with limb–spine coupling disabled."""
    log_dir = 'logs/ex3_no_limb_spine_coupling'
    sim_parameters = _walk_params(
        timestep,
        disable_limb_spine_coupling=True,
    )
    _run_walk_sim(
        log_dir, sim_parameters, timestep,
        record=True,
        record_name='ex3_2_no_limb_spine_coupling.mp4',
        headless=False,
        fast=False,
    )
    speed, _, cot = metrics_from_log_dir(log_dir)
    pylog.info(
        'No limb-spine coupling: speed=%.4f m/s, CoT=%.4f',
        speed, cot,
    )


def exercise_3_limb_spine_antiphase(timestep, phase_offset=None):
    """Exercise 3.3b: walk with limb–spine phase at anti-phase to optimal."""
    if phase_offset is None:
        phase_offset = _load_optimal_phase_offset(default=np.pi)
    anti_phase = float(phase_offset) + np.pi
    log_dir = 'logs/ex3_antiphase_walk'
    sim_parameters = _walk_params(
        timestep,
        limb_body_phase_offset=anti_phase,
    )
    _run_walk_sim(
        log_dir, sim_parameters, timestep,
        record=True,
        record_name='ex3_3b_antiphase_walk.mp4',
        headless=False,
        fast=False,
    )


def exercise_3a_coordination(
        timestep,
        run_sweep=True,
        run_videos=True,
        drive_values=None,
        phase_values=None,
        processes=4,
):
    """Exercise 3.3: 2D sweep of limb–body phase offset vs drive."""
    if drive_values is None:
        drive_values = np.linspace(1.5, 2.5, 5)
    if phase_values is None:
        phase_values = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    sweep_dir = 'logs/ex3a_sweep'
    os.makedirs(sweep_dir, exist_ok=True)

    if run_sweep:
        parameter_set = []
        sim_args = []
        idx = 0
        for drive in drive_values:
            for phase_off in phase_values:
                sim_parameters = _walk_params(
                    timestep,
                    duration=SWEEP_DURATION,
                    drive=float(drive),
                    limb_body_phase_offset=float(phase_off),
                )
                parameter_set.append(sim_parameters)
                out = os.path.join(
                    sweep_dir,
                    f'd{drive:.2f}_psi{phase_off:.2f}',
                )
                sim_args.append({
                    'sim_parameters': sim_parameters,
                    'arena': 'land',
                    'fast': True,
                    'headless': True,
                    'output': out,
                    'verbose': False,
                })
                idx += 1
        pylog.info('Running %d simulations for Ex3a sweep', len(sim_args))
        simulation_sweep(sim_args, processes=processes)

    speed_map = np.zeros((len(phase_values), len(drive_values)))
    cot_map = np.zeros_like(speed_map)

    for i_p, phase_off in enumerate(phase_values):
        for i_d, drive in enumerate(drive_values):
            out = os.path.join(
                sweep_dir, f'd{drive:.2f}_psi{phase_off:.2f}',
            )
            try:
                speed, _, cot = metrics_from_log_dir(out)
                speed_map[i_p, i_d] = speed
                cot_map[i_p, i_d] = cot
            except Exception as exc:
                pylog.warning('Missing metrics for %s: %s', out, exc)
                speed_map[i_p, i_d] = np.nan
                cot_map[i_p, i_d] = np.nan

    plot_sweep_heatmaps(
        drive_values,
        phase_values,
        speed_map,
        cot_map,
        'Drive',
        'Limb–body phase offset [rad]',
        'results',
        'ex3a',
    )

    # Optimal phase: max speed, tie-break min CoT
    valid = np.isfinite(speed_map)
    if np.any(valid):
        best_flat = np.nanargmax(speed_map)
        i_p, i_d = np.unravel_index(best_flat, speed_map.shape)
        # Refine among top speeds
        top_speed = np.nanmax(speed_map)
        candidates = np.where(speed_map >= top_speed - 1e-4)
        if len(candidates[0]) > 1:
            cots = cot_map[candidates]
            best_c = np.argmin(cots)
            i_p = candidates[0][best_c]
            i_d = candidates[1][best_c]
        opt_phase = float(phase_values[i_p])
        opt_drive = float(drive_values[i_d])
        os.makedirs(os.path.dirname(OPTIMAL_PHASE_FILE), exist_ok=True)
        with open(OPTIMAL_PHASE_FILE, 'wb') as f:
            pickle.dump({
                'limb_body_phase_offset': opt_phase,
                'drive': opt_drive,
                'speed': float(speed_map[i_p, i_d]),
                'cot': float(cot_map[i_p, i_d]),
            }, f)
        pylog.info(
            'Optimal: drive=%.2f, phase=%.2f rad, speed=%.4f, CoT=%.4f',
            opt_drive, opt_phase, speed_map[i_p, i_d], cot_map[i_p, i_d],
        )
    else:
        opt_phase = 0.0
        opt_drive = WALK_DRIVE

    if run_videos:
        log_ideal = 'logs/ex3_ideal_walk'
        _run_walk_sim(
            log_ideal,
            _walk_params(
                timestep,
                drive=opt_drive,
                limb_body_phase_offset=opt_phase,
            ),
            timestep,
            record=True,
            record_name='ex3_3a_ideal_walk.mp4',
            headless=False,
            fast=False,
        )
        exercise_3_limb_spine_antiphase(timestep, phase_offset=opt_phase)

    return opt_phase, opt_drive, speed_map, cot_map


def exercise_3b_coordination(
        timestep,
        run_sweep=True,
        run_videos=True,
        body_gains=None,
        limb_gains=None,
        processes=4,
):
    """Exercise 3.4: 2D sweep of axial and limb amplitude gains."""
    opt_phase = _load_optimal_phase_offset(default=0.0)
    opt_data = {}
    if os.path.isfile(OPTIMAL_PHASE_FILE):
        with open(OPTIMAL_PHASE_FILE, 'rb') as f:
            opt_data = pickle.load(f)
    opt_drive = opt_data.get('drive', WALK_DRIVE)

    if body_gains is None:
        body_gains = np.linspace(0.0, 1.5, 7)
    if limb_gains is None:
        limb_gains = np.linspace(0.0, 1.5, 7)

    sweep_dir = 'logs/ex3b_sweep'
    os.makedirs(sweep_dir, exist_ok=True)

    if run_sweep:
        sim_args = []
        for body_g in body_gains:
            for limb_g in limb_gains:
                sim_parameters = _walk_params(
                    timestep,
                    duration=SWEEP_DURATION,
                    drive=opt_drive,
                    limb_body_phase_offset=opt_phase,
                    body_amplitude_gain=float(body_g),
                    limb_amplitude_gain=float(limb_g),
                )
                out = os.path.join(
                    sweep_dir,
                    f'bg{body_g:.2f}_lg{limb_g:.2f}',
                )
                sim_args.append({
                    'sim_parameters': sim_parameters,
                    'arena': 'land',
                    'fast': True,
                    'headless': True,
                    'output': out,
                    'verbose': False,
                })
        pylog.info('Running %d simulations for Ex3b sweep', len(sim_args))
        simulation_sweep(sim_args, processes=processes)

    speed_map = np.zeros((len(limb_gains), len(body_gains)))
    cot_map = np.zeros_like(speed_map)

    for i_l, limb_g in enumerate(limb_gains):
        for i_b, body_g in enumerate(body_gains):
            out = os.path.join(
                sweep_dir, f'bg{body_g:.2f}_lg{limb_g:.2f}',
            )
            try:
                speed, _, cot = metrics_from_log_dir(out)
                speed_map[i_l, i_b] = speed
                cot_map[i_l, i_b] = cot
            except Exception as exc:
                pylog.warning('Missing metrics for %s: %s', out, exc)
                speed_map[i_l, i_b] = np.nan
                cot_map[i_l, i_b] = np.nan

    plot_sweep_heatmaps(
        body_gains,
        limb_gains,
        speed_map,
        cot_map,
        'Body amplitude gain',
        'Limb amplitude gain',
        'results',
        'ex3b',
    )

    if run_videos and np.any(np.isfinite(speed_map)):
        best_flat = np.nanargmax(speed_map)
        i_l, i_b = np.unravel_index(best_flat, speed_map.shape)
        log_dir = 'logs/ex3b_optimal_amplitudes'
        _run_walk_sim(
            log_dir,
            _walk_params(
                timestep,
                drive=opt_drive,
                limb_body_phase_offset=opt_phase,
                body_amplitude_gain=float(body_gains[i_b]),
                limb_amplitude_gain=float(limb_gains[i_l]),
            ),
            timestep,
            record=True,
            record_name='ex3_4_optimal_amplitudes.mp4',
            headless=False,
            fast=False,
        )

    return speed_map, cot_map


def _load_optimal_phase_offset(default=0.0):
    if os.path.isfile(OPTIMAL_PHASE_FILE):
        with open(OPTIMAL_PHASE_FILE, 'rb') as f:
            return pickle.load(f).get('limb_body_phase_offset', default)
    return default


if __name__ == '__main__':
    timestep = 5e-3
    exercise_3_spine_analysis(timestep)
    exercise_3_disable_limb_spine_coupling(timestep)
    exercise_3a_coordination(timestep, run_sweep=True, run_videos=True)
    exercise_3b_coordination(timestep, run_sweep=True, run_videos=True)

    if not save_plots():
        plt.show()
    else:
        save_figures()
