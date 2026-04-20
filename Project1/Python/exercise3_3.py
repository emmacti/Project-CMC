#!/usr/bin/env python3
"""
Exercise 3.3

Two controller setups (Tab. 3):
  • Combined : w_ipsi=W_IPSI  +  full ipsilateral coupling
  • Decoupled: w_ipsi=W_IPSI  +  ipsilateral coupling weights = 0

Three disruption types (Tab. 2), five probabilities [0 … 15 %]:
  • Muted Sensors       : disruption_p_sensors in [0,15%], disruption_p_couplings=0
  • Removed Couplings   : disruption_p_sensors=0,          disruption_p_couplings in [0,15%]
  • Mixed               : both = disruption_p in [0,15%]

Preliminary "what-do-you-observe" run: mixed disruption at 20% (single seed).

For reproducibility each (setup, disruption type, probability) is run with
N_SEEDS seeds and results are averaged.
"""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    filter_signals,
    LINKS_MASSES,
)
from simulate import runsim, run_multiple

pylog.set_level('warning')

# paths 
BASE_PATH  = 'logs/exercise3_3/'
PLOT_PATH  = 'results'
os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(BASE_PATH,  exist_ok=True)

# CPG parameters (Q3.3 sets offset_amp = 0) 
DRIVE_LEFT  = 3
DRIVE_RIGHT = 3
DRIVE_LOW   = 1
DRIVE_HIGH  = 5
A_RATE      = np.ones(8) * 3
OFFSET_FREQ = np.ones(8) * 1
OFFSET_AMP  = np.ones(8) * 0          # <-- Q3.3: set to 0
G_FREQ      = np.ones(8) * 0.5
G_AMP       = np.ones(8) * 0.25
PHASELAG    = np.ones(7) * np.pi * 2 / 8
CW_ROSTRAL  = 5
CW_CAUDAL   = 5
CW_CONTRA   = 10
INIT_PHASE  = np.random.default_rng(seed=42).uniform(0.0, 2 * np.pi, size=16)
W_IPSI      = 10.0

# experiment design 
N_STEPS      = 10001          # 40 s simulation
BUFFER_SIZE  = 10001
SKIP_START   = 500            # drop initial transient samples
MAX_WORKERS  = 8

# Five disruption probabilities 0 … 15 % (inclusive)
DISRUPTION_PROBS = np.linspace(0.0, 0.15, 5)

# Seeds for averaging
N_SEEDS = 3
SEEDS   = [42, 123, 7]

# Preliminary mixed-disruption observation run
OBS_P = 0.20   # 20 % mixed disruption


# Helpers

def _base_config(w_ipsi=W_IPSI, coupling_rostral=CW_ROSTRAL,
                 coupling_caudal=CW_CAUDAL,
                 disruption_p_sensors=0.0, disruption_p_couplings=0.0,
                 random_seed=42):
    return {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left':  DRIVE_LEFT,
            'drive_right': DRIVE_RIGHT,
            'd_low':  DRIVE_LOW,
            'd_high': DRIVE_HIGH,
            'a_rate':      A_RATE.copy(),
            'offset_freq': OFFSET_FREQ.copy(),
            'offset_amp':  OFFSET_AMP.copy(),
            'G_freq':      G_FREQ.copy(),
            'G_amp':       G_AMP.copy(),
            'PL':          PHASELAG.copy(),
            'coupling_weights_rostral': coupling_rostral,
            'coupling_weights_caudal':  coupling_caudal,
            'coupling_weights_contra':  CW_CONTRA,
            'init_phase':  INIT_PHASE.copy(),
            'w_ipsi':      w_ipsi,
            'disruption_p_sensors':   disruption_p_sensors,
            'disruption_p_couplings': disruption_p_couplings,
            'random_seed': random_seed,
        }
    }


def load_sim_data(hdf5_path, skip_start=SKIP_START):
    """Load simulation sensor data, drop initial transient."""
    with h5py.File(hdf5_path, 'r') as f:
        sim_times    = f['times'][:]
        sensor_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_joints= f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    sim_times        = sim_times[skip_start:]
    links_pos        = sensor_links[skip_start:, :, 7:10]
    links_vel        = sensor_links[skip_start:, :, 14:17]
    joints_pos       = sensor_joints[skip_start:, :, 0]
    joints_vel       = sensor_joints[skip_start:, :, 1]
    joints_torques   = sensor_joints[skip_start:, :, 2]
    return sim_times, links_pos, links_vel, joints_pos, joints_vel, joints_torques


def get_speed_cot(hdf5_path):
    t, lp, lv, _, jv, jt = load_sim_data(hdf5_path)
    speed, _ = compute_mechanical_speed(links_positions=lp, links_velocities=lv)
    _, cot   = compute_mechanical_energy_and_cot(
        times=t, links_positions=lp,
        joints_torques=jt, joints_velocities=jv)
    return float(speed), float(cot)


def com_xy(links_positions):
    n        = links_positions.shape[1]
    masses   = LINKS_MASSES[:n]
    mass_sum = float(masses.sum())
    com      = (links_positions * masses[None, :, None]).sum(axis=1) / mass_sum
    return com[:, :2]


def _run_tag(setup_name, disruption_type, p, seed):
    """Build a unique filename token for one simulation."""
    p_str = f'{p:.4f}'.replace('.', 'p')
    return f'{setup_name}_{disruption_type}_p{p_str}_seed{seed}'


def _hdf5_path(tag):
    return os.path.join(BASE_PATH, f'simulation_{tag}.hdf5')


def _ctl_path(tag):
    return os.path.join(BASE_PATH, f'controller_{tag}.pkl')


# Part A – Preliminary observation: 20 % mixed disruption

def run_observation():
    """
    Run three simulations (no disruption, combined+20%, decoupled+20%) and
    produce joint-angle and CoM-trajectory comparison plots.
    """
    print("\n── Part A: 20 % mixed disruption observation ─────────────────────")

    obs_cases = {
        'no_disruption':    dict(w_ipsi=W_IPSI,  coupling_rostral=CW_ROSTRAL,
                                 coupling_caudal=CW_CAUDAL,
                                 p_s=0.0,    p_c=0.0),
        'combined_20pct':   dict(w_ipsi=W_IPSI,  coupling_rostral=CW_ROSTRAL,
                                 coupling_caudal=CW_CAUDAL,
                                 p_s=OBS_P,  p_c=OBS_P),
        'decoupled_20pct':  dict(w_ipsi=W_IPSI,  coupling_rostral=0.0,
                                 coupling_caudal=0.0,
                                 p_s=OBS_P,  p_c=OBS_P),
    }

    for name, cfg in obs_cases.items():
        tag      = f'obs_{name}'
        hdf5_p   = _hdf5_path(tag)
        if os.path.isfile(hdf5_p):
            print(f'  [cached] {name}')
            continue
        print(f'  Running {name} ...')
        controller = _base_config(
            w_ipsi=cfg['w_ipsi'],
            coupling_rostral=cfg['coupling_rostral'],
            coupling_caudal=cfg['coupling_caudal'],
            disruption_p_sensors=cfg['p_s'],
            disruption_p_couplings=cfg['p_c'],
            random_seed=SEEDS[0],
        )
        runsim(
            controller=controller,
            base_path=BASE_PATH,
            hdf5_name=f'simulation_obs_{name}.hdf5',
            controller_name=f'controller_obs_{name}.pkl',
            runtime_n_iterations=N_STEPS,
            runtime_buffer_size=BUFFER_SIZE,
            fast=True, headless=True,
        )

    # Plot: joint angles (first 10 s after transient) 
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.5, 8))
    T_SHOW = 10.0

    for ax, name in zip(axes, obs_cases):
        hdf5_p = _hdf5_path(f'obs_{name}')
        if not os.path.isfile(hdf5_p):
            ax.set_title(f'{name}\n(not found)')
            continue
        t, lp, lv, jp, jv, jt = load_sim_data(hdf5_p)
        mask = t <= (t[0] + T_SHOW)
        for j in range(min(8, jp.shape[1])):
            ax.plot(t[mask], jp[mask, j], color=colors[j], lw=0.9,
                    label=f'j{j}')
        ax.set_title(name.replace('_', ' '), fontsize=9)
        ax.set_xlabel('time (s)')
        if ax is axes[0]:
            ax.set_ylabel('joint angle (rad)')
        ax.legend(fontsize=6, ncol=2)

    fig.suptitle('Q3.3 – Joint angles: no disruption vs 20% mixed disruption (first 10 s)',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'ex3_3_obs_joint_angles.png'), dpi=200)
    plt.close()

    # Plot: CoM trajectories 
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, obs_cases):
        hdf5_p = _hdf5_path(f'obs_{name}')
        if not os.path.isfile(hdf5_p):
            ax.set_title(f'{name}\n(not found)')
            continue
        t, lp, _, _, _, _ = load_sim_data(hdf5_p)
        xy = com_xy(lp)
        ax.plot(xy[:, 0], xy[:, 1], lw=1.5)
        ax.set_title(name.replace('_', ' '), fontsize=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.axis('equal')

    fig.suptitle('Q3.3 – CoM trajectory: no disruption vs 20% mixed disruption',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'ex3_3_obs_com_trajectory.png'), dpi=200)
    plt.close()

    # ── Print metric comparison ───────────────────────────────────────────
    print(f"\n  {'Case':<25} {'Speed (m/s)':>12} {'CoT (J/m)':>12}")
    print('  ' + '-'*51)
    for name in obs_cases:
        hdf5_p = _hdf5_path(f'obs_{name}')
        if not os.path.isfile(hdf5_p):
            print(f'  {name:<25} {"N/A":>12} {"N/A":>12}')
            continue
        spd, cot = get_speed_cot(hdf5_p)
        print(f'  {name:<25} {spd:>12.4f} {cot:>12.4f}')



# Part B – Ablation sweep (Tab. 2 × Tab. 3, averaged over N_SEEDS seeds)

# Setups (Tab. 3)
SETUPS = {
    'combined':  dict(coupling_rostral=CW_ROSTRAL, coupling_caudal=CW_CAUDAL),
    'decoupled': dict(coupling_rostral=0.0,         coupling_caudal=0.0),
}

# Disruption types (Tab. 2)
DISRUPTION_TYPES = {
    'muted_sensors':     lambda p: dict(p_s=p,   p_c=0.0),
    'removed_couplings': lambda p: dict(p_s=0.0, p_c=p),
    'mixed':             lambda p: dict(p_s=p,   p_c=p),
}


def run_ablation_sweep():
    """Run all (setup × disruption_type × probability × seed) simulations."""
    print("\n── Part B: ablation sweep ───────────────────────────────────────")

    sim_list = []   # collect configs for run_multiple

    for setup_name, setup_cfg in SETUPS.items():
        for dtype, dtype_fn in DISRUPTION_TYPES.items():
            for p in DISRUPTION_PROBS:
                for seed in SEEDS:
                    tag    = _run_tag(setup_name, dtype, p, seed)
                    hdf5_p = _hdf5_path(tag)
                    if os.path.isfile(hdf5_p):
                        continue   # already done

                    d_cfg  = dtype_fn(p)
                    config = _base_config(
                        w_ipsi=W_IPSI,
                        coupling_rostral=setup_cfg['coupling_rostral'],
                        coupling_caudal=setup_cfg['coupling_caudal'],
                        disruption_p_sensors=d_cfg['p_s'],
                        disruption_p_couplings=d_cfg['p_c'],
                        random_seed=seed,
                    )
                    sim_list.append({
                        'controller': config,
                        'hdf5_name': f'simulation_{tag}.hdf5',
                        'controller_name': f'controller_{tag}.pkl',
                        'runtime_n_iterations': N_STEPS,
                        'runtime_buffer_size':  BUFFER_SIZE,
                        'fast': True, 'headless': True,
                    })

    if sim_list:
        total = len(sim_list)
        print(f'  Launching {total} simulations (max_workers={MAX_WORKERS}) ...')
        # run_multiple expects a parameter_grid interface; we run individually
        # to stay compatible with any run_multiple signature.
        for i, s in enumerate(sim_list):
            print(f'  [{i+1}/{total}] {s["hdf5_name"]}')
            runsim(
                controller=s['controller'],
                base_path=BASE_PATH,
                hdf5_name=s['hdf5_name'],
                controller_name=s['controller_name'],
                runtime_n_iterations=s['runtime_n_iterations'],
                runtime_buffer_size=s['runtime_buffer_size'],
                fast=s['fast'],
                headless=s['headless'],
            )
    else:
        print('  All simulations already cached.')


def collect_ablation_metrics():
    """
    Returns nested dict:
      results[setup_name][dtype][p_idx] = {'speed': [...], 'cot': [...]}
    averaged over seeds.
    """
    results = {}
    for setup_name in SETUPS:
        results[setup_name] = {}
        for dtype in DISRUPTION_TYPES:
            results[setup_name][dtype] = {
                'speed_mean': np.zeros(len(DISRUPTION_PROBS)),
                'speed_std':  np.zeros(len(DISRUPTION_PROBS)),
                'cot_mean':   np.zeros(len(DISRUPTION_PROBS)),
                'cot_std':    np.zeros(len(DISRUPTION_PROBS)),
            }
            for i_p, p in enumerate(DISRUPTION_PROBS):
                speeds, cots = [], []
                for seed in SEEDS:
                    tag    = _run_tag(setup_name, dtype, p, seed)
                    hdf5_p = _hdf5_path(tag)
                    if not os.path.isfile(hdf5_p):
                        continue
                    spd, cot = get_speed_cot(hdf5_p)
                    speeds.append(spd)
                    cots.append(cot)
                if speeds:
                    results[setup_name][dtype]['speed_mean'][i_p] = np.mean(speeds)
                    results[setup_name][dtype]['speed_std'][i_p]  = np.std(speeds)
                    results[setup_name][dtype]['cot_mean'][i_p]   = np.mean(cots)
                    results[setup_name][dtype]['cot_std'][i_p]    = np.std(cots)
    return results


def plot_ablation(results):
    """
    One figure per metric (speed / CoT):
      2 rows  = combined / decoupled
      3 cols  = muted sensors / removed couplings / mixed
    """
    dtype_labels = {
        'muted_sensors':     'Muted Sensors',
        'removed_couplings': 'Removed Couplings',
        'mixed':             'Mixed',
    }
    setup_labels = {
        'combined':  'Combined (SF + coupling)',
        'decoupled': 'Decoupled (SF only)',
    }
    p_pct = DISRUPTION_PROBS * 100

    for metric_key, metric_label, fname_suffix in [
        ('speed', 'Forward speed (m/s)', 'speed'),
        ('cot',   'CoT (J/m)',            'cot'),
    ]:
        fig, axes = plt.subplots(
            2, 3, figsize=(14, 7),
            sharex=True,
        )
        for r, setup_name in enumerate(SETUPS):
            for c, dtype in enumerate(DISRUPTION_TYPES):
                ax  = axes[r, c]
                d   = results[setup_name][dtype]
                mu  = d[f'{metric_key}_mean']
                std = d[f'{metric_key}_std']

                ax.plot(p_pct, mu, 'o-', color='tab:blue', lw=1.8)
                ax.fill_between(p_pct, mu - std, mu + std,
                                alpha=0.25, color='tab:blue')
                ax.axhline(mu[0], color='k', ls=':', lw=0.8,
                           label='no disruption')
                ax.set_title(
                    f'{setup_labels[setup_name]}\n{dtype_labels[dtype]}',
                    fontsize=8)
                ax.set_xlabel('Disruption prob. (%)')
                if c == 0:
                    ax.set_ylabel(metric_label)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.25)

        fig.suptitle(
            f'Q3.3 – {metric_label} vs disruption probability\n'
            f'(mean ± std over {N_SEEDS} seeds)',
            fontsize=11)
        plt.tight_layout()
        outf = os.path.join(PLOT_PATH, f'ex3_3_ablation_{fname_suffix}.png')
        plt.savefig(outf, dpi=200)
        plt.close()
        print(f'  Saved: {outf}')

    # ── Overlay: combined vs decoupled for each disruption type ──────────
    for metric_key, metric_label, fname_suffix in [
        ('speed', 'Forward speed (m/s)', 'speed'),
        ('cot',   'CoT (J/m)',            'cot'),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        setup_colors = {'combined': 'tab:blue', 'decoupled': 'tab:orange'}

        for c, dtype in enumerate(DISRUPTION_TYPES):
            ax = axes[c]
            for setup_name in SETUPS:
                d   = results[setup_name][dtype]
                mu  = d[f'{metric_key}_mean']
                std = d[f'{metric_key}_std']
                col = setup_colors[setup_name]
                ax.plot(p_pct, mu, 'o-', color=col,
                        lw=1.8, label=setup_labels[setup_name])
                ax.fill_between(p_pct, mu - std, mu + std,
                                alpha=0.2, color=col)
            ax.set_title(dtype_labels[dtype], fontsize=9)
            ax.set_xlabel('Disruption prob. (%)')
            if c == 0:
                ax.set_ylabel(metric_label)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)

        fig.suptitle(
            f'Q3.3 – {metric_label}: combined vs decoupled\n'
            f'(mean ± std over {N_SEEDS} seeds)',
            fontsize=11)
        plt.tight_layout()
        outf = os.path.join(PLOT_PATH,
                            f'ex3_3_ablation_{fname_suffix}_overlay.png')
        plt.savefig(outf, dpi=200)
        plt.close()
        print(f'  Saved: {outf}')



# Main entry point

def exercise3_3(**kwargs):
    """ex3.3 main"""
    fast     = kwargs.pop('fast', False)
    headless = kwargs.pop('headless', False)
    plot     = kwargs.pop('plot', False)

    # Part A: qualitative observation at 20% mixed disruption 
    run_observation()

    # Part B: full ablation sweep 
    run_ablation_sweep()

    # Analysis & plots 
    print("\n── Collecting metrics and plotting ─────────────────────────────")
    results = collect_ablation_metrics()
    plot_ablation(results)

    print("\nAll Q3.3 results saved to:", PLOT_PATH)

    if plot:
        plt.show()


if __name__ == '__main__':
    exercise3_3(plot=True)

