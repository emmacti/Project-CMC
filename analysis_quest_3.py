#!/usr/bin/env python3
"""
Post-processing and analysis for Questions 3.1 and 3.2.

Q3.1: Compare CPG with stretch feedback (w_ipsi=3.0) vs without (w_ipsi=0).
      Plots oscillator states (theta, r), muscle output sum/diff, joint angles,
      CoM trajectory. Computes neural frequencies, amplitudes, forward speed, CoT.

Q3.2: Sweep w_ipsi in [-3.0, 17.0], plot CoT, speed, neural freq, neural amp vs w_ipsi.
"""

import os
import pickle
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# imports
from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    compute_neural_phase_lags,
    filter_signals,
    LINKS_MASSES,
)

# paths
BASE_PATH_31 = 'logs/exercise3_1/'
BASE_PATH_32 = 'logs/exercise3_2/'
PLOT_PATH    = 'results'
os.makedirs(PLOT_PATH, exist_ok=True)

# w_ipsi used in the "with feedback" run (match exercise3_1.py)
W_IPSI_31 = 3.0

# Transient skip (seconds) – skip early cycles before steady state
TRANSIENT_S = 2.0


# Helper utilities

def load_sim(hdf5_path):
    """Load times, link sensor data, joint sensor data from HDF5."""
    with h5py.File(hdf5_path, 'r') as f:
        times            = f['times'][:]
        sensor_links     = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_joints    = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    links_positions  = sensor_links[:, :, 7:10]
    links_velocities = sensor_links[:, :, 14:17]
    joints_positions = sensor_joints[:, :, 0]
    joints_velocities= sensor_joints[:, :, 1]
    joints_torques   = sensor_joints[:, :, 2]
    return (times, links_positions, links_velocities,
            joints_positions, joints_velocities, joints_torques)


def load_controller(pkl_path):
    """Load controller pickle; return state array, n_osc, n_body_joints."""
    with open(pkl_path, 'rb') as f:
        ctrl = pickle.load(f)
    state = ctrl['state']           # (T, 3*n_osc)
    n_total = state.shape[1]
    assert n_total % 3 == 0, f"Unexpected controller state width {n_total}"
    n_osc         = n_total // 3
    n_body_joints = n_osc // 2
    return state, n_osc, n_body_joints


def unpack_controller(state, n_osc):
    """Split controller state into phases, amplitudes, motor_storage."""
    phases        = state[:, :n_osc]
    amplitudes    = state[:, n_osc:2*n_osc]
    motor_storage = state[:, 2*n_osc:3*n_osc]
    motor_left    = motor_storage[:, 0::2]   # (T, n_body_joints)
    motor_right   = motor_storage[:, 1::2]
    return phases, amplitudes, motor_left, motor_right


def steady_mask(times, transient_s=TRANSIENT_S):
    """Boolean mask discarding the first `transient_s` seconds."""
    return times >= (times[0] + transient_s)


def compute_all_metrics(times, links_positions, links_velocities,
                        joints_positions, joints_velocities, joints_torques,
                        motor_left, motor_right, transient_s=TRANSIENT_S):
    """Return dict of scalar performance metrics (computed on steady-state window)."""
    mask = steady_mask(times, transient_s)
    t_ss  = times[mask]

    # --- mechanical metrics ---
    speed_fwd, speed_lat = compute_mechanical_speed(
        links_positions=links_positions[mask],
        links_velocities=links_velocities[mask],
    )
    energy, cot = compute_mechanical_energy_and_cot(
        times=t_ss,
        links_positions=links_positions[mask],
        joints_torques=joints_torques[mask],
        joints_velocities=joints_velocities[mask],
    )

    # --- neural metrics ---
    neural_diff = motor_left[mask] - motor_right[mask]   # (T_ss, n_joints)
    smooth      = filter_signals(times=t_ss, signals=neural_diff)
    freqs, _, amps = compute_frequency_amplitude_fft(times=t_ss, smooth_signals=smooth)

    n_joints      = neural_diff.shape[1]
    inds_couples  = [[i, i+1] for i in range(n_joints - 1)]
    _, ipl_mean   = compute_neural_phase_lags(
        times=t_ss, smooth_signals=smooth, freqs=freqs, inds_couples=inds_couples)

    return {
        'speed_fwd':  float(speed_fwd),
        'speed_lat':  float(speed_lat),
        'energy':     float(energy),
        'cot':        float(cot),
        'freq':       freqs,           # array (n_joints,)
        'amp':        amps,            # array (n_joints,)
        'ipl_mean':   float(ipl_mean),
    }


def com_xy(links_positions):
    n = links_positions.shape[1]
    masses   = LINKS_MASSES[:n]
    mass_sum = float(masses.sum())
    com      = (links_positions * masses[None, :, None]).sum(axis=1) / mass_sum
    return com[:, :2]


# Q3.1 – with-SF vs without-SF comparison

def analysis_3_1():
    """
    Load simulation results for w_ipsi = W_IPSI_31 (with SF) and w_ipsi = 0
    (without SF).  Produce all required plots and print metrics.
    """
    print("\n" + "="*70)
    print("Q3.1 – Sensory Feedback Analysis")
    print("="*70)

    cases = {
        f'With SF (w_ipsi={W_IPSI_31})': {
            'hdf5': BASE_PATH_31 + 'simulation_with_sf.hdf5',
            'pkl':  BASE_PATH_31 + 'controller_with_sf.pkl',
            'color': 'tab:blue',
            'ls':    '-',
        },
        'Without SF (w_ipsi=0)': {
            'hdf5': BASE_PATH_31 + 'simulation_without_sf.hdf5',
            'pkl':  BASE_PATH_31 + 'controller_without_sf.pkl',
            'color': 'tab:orange',
            'ls':    '--',
        },
    }

    data = {}
    for label, cfg in cases.items():
        if not os.path.isfile(cfg['hdf5']):
            print(f"  [SKIP] {label}: {cfg['hdf5']} not found.")
            continue
        print(f"\nLoading {label} ...")
        (times, lp, lv, jp, jv, jt) = load_sim(cfg['hdf5'])
        state, n_osc, n_body = load_controller(cfg['pkl'])
        phases, amplitudes, ml, mr = unpack_controller(state, n_osc)
        metrics = compute_all_metrics(times, lp, lv, jp, jv, jt, ml, mr)

        print(f"  Forward speed : {metrics['speed_fwd']:.4f} m/s")
        print(f"  Lateral speed : {metrics['speed_lat']:.4f} m/s")
        print(f"  Energy        : {metrics['energy']:.4f} J")
        print(f"  CoT           : {metrics['cot']:.4f} J/m")
        print(f"  Mean IPL_neur : {metrics['ipl_mean']:.4f} rad")
        print(f"  Neural freq   : {np.round(metrics['freq'], 3)}")
        print(f"  Neural amp    : {np.round(metrics['amp'],  3)}")

        data[label] = dict(cfg=cfg, times=times, lp=lp, lv=lv,
                           jp=jp, jv=jv, jt=jt,
                           phases=phases, amplitudes=amplitudes,
                           ml=ml, mr=mr, metrics=metrics,
                           n_osc=n_osc, n_body=n_body)

    if len(data) < 1:
        print("No data found – skipping Q3.1 plots.")
        return

    # determine common time window for plots 
    T_PLOT = 5.0   # seconds to show

    # 1. Oscillator phases (theta) 
    fig, axes = plt.subplots(1, len(data), figsize=(7*len(data), 4), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (label, d) in zip(axes, data.items()):
        mask = d['times'] <= (d['times'][0] + T_PLOT)
        t    = d['times'][mask]
        n_show = min(3, d['n_body'])
        for j in range(n_show):
            ax.plot(t, d['phases'][mask, 2*j],   label=f'θ L j{j}')
            ax.plot(t, d['phases'][mask, 2*j+1], ls='--', label=f'θ R j{j}')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('θ (rad)')
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle('Q3.1 – Oscillator phases (first 5 s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'ex3_1_theta.png'), dpi=200)
    plt.close()

    # 2. Oscillator amplitudes (r) 
    fig, axes = plt.subplots(1, len(data), figsize=(7*len(data), 4), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (label, d) in zip(axes, data.items()):
        mask  = d['times'] <= (d['times'][0] + T_PLOT)
        t     = d['times'][mask]
        n_show = min(3, d['n_body'])
        for j in range(n_show):
            ax.plot(t, d['amplitudes'][mask, 2*j],   label=f'r L j{j}')
            ax.plot(t, d['amplitudes'][mask, 2*j+1], ls='--', label=f'r R j{j}')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('r (amplitude)')
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle('Q3.1 – Oscillator amplitudes (first 5 s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'ex3_1_r.png'), dpi=200)
    plt.close()

    # 3. Muscle output sum (ML + MR) 
    fig, axes = plt.subplots(1, len(data), figsize=(7*len(data), 4), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (label, d) in zip(axes, data.items()):
        mask   = d['times'] <= (d['times'][0] + T_PLOT)
        t      = d['times'][mask]
        n_show = min(3, d['n_body'])
        msum   = d['ml'] + d['mr']
        for j in range(n_show):
            ax.plot(t, msum[mask, j], label=f'ML+MR j{j}')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('ML+MR')
        ax.legend(fontsize=7)
    fig.suptitle('Q3.1 – Muscle output sum (first 5 s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'ex3_1_muscle_sum.png'), dpi=200)
    plt.close()

    # 4. Muscle output difference (ML - MR) 
    fig, axes = plt.subplots(1, len(data), figsize=(7*len(data), 4), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (label, d) in zip(axes, data.items()):
        mask   = d['times'] <= (d['times'][0] + T_PLOT)
        t      = d['times'][mask]
        n_show = min(3, d['n_body'])
        mdiff  = d['ml'] - d['mr']
        for j in range(n_show):
            ax.plot(t, mdiff[mask, j], label=f'ML-MR j{j}')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('ML−MR')
        ax.legend(fontsize=7)
    fig.suptitle('Q3.1 – Muscle output difference (first 5 s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'ex3_1_muscle_diff.png'), dpi=200)
    plt.close()

    # 5. Body joint angles 
    fig, axes = plt.subplots(1, len(data), figsize=(7*len(data), 4), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (label, d) in zip(axes, data.items()):
        mask   = d['times'] <= (d['times'][0] + T_PLOT)
        t      = d['times'][mask]
        n_show = min(5, d['jp'].shape[1])
        for j in range(n_show):
            ax.plot(t, d['jp'][mask, j], label=f'joint {j}')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('joint angle (rad)')
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle('Q3.1 – Body joint angles (first 5 s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'ex3_1_joint_angles.png'), dpi=200)
    plt.close()

    # 6. CoM trajectories (side by side + overlay) 
    fig, axes = plt.subplots(1, len(data), figsize=(5*len(data), 5))
    if len(data) == 1:
        axes = [axes]
    for ax, (label, d) in zip(axes, data.items()):
        xy = com_xy(d['lp'])
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2, color=d['cfg']['color'])
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.axis('equal')
    fig.suptitle('Q3.1 – CoM trajectory (XY)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'ex3_1_com_trajectory.png'), dpi=200)
    plt.close()

    # 7. Metric comparison bar chart 
    if len(data) == 2:
        labels_list = list(data.keys())
        m0 = data[labels_list[0]]['metrics']
        m1 = data[labels_list[1]]['metrics']

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        metric_pairs = [
            ('speed_fwd', 'Forward speed (m/s)'),
            ('cot',       'CoT (J/m)'),
            ('freq',      'Mean neural freq (Hz)'),
            ('amp',       'Mean neural amp'),
        ]
        colors_bar = [data[l]['cfg']['color'] for l in labels_list]
        for ax, (key, ylabel) in zip(axes, metric_pairs):
            vals = []
            for l in labels_list:
                v = data[l]['metrics'][key]
                vals.append(float(np.mean(v)))
            bars = ax.bar(labels_list, vals, color=colors_bar, edgecolor='k', width=0.4)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontsize=9)
            ax.tick_params(axis='x', labelrotation=15, labelsize=8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        fig.suptitle('Q3.1 – Metric comparison: with vs without stretch feedback',
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, 'ex3_1_metric_comparison.png'), dpi=200)
        plt.close()

        # 8. Per-joint freq and amp overlay 
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for label, d in data.items():
            m   = d['metrics']
            cfg = d['cfg']
            j_idx = np.arange(len(m['freq']))
            axes[0].plot(j_idx, m['freq'], marker='o', color=cfg['color'],
                         ls=cfg['ls'], label=label)
            axes[1].plot(j_idx, m['amp'],  marker='o', color=cfg['color'],
                         ls=cfg['ls'], label=label)
        axes[0].set_xlabel('Joint index')
        axes[0].set_ylabel('Neural frequency (Hz)')
        axes[0].set_title('Neural frequency per joint')
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel('Joint index')
        axes[1].set_ylabel('Neural amplitude')
        axes[1].set_title('Neural amplitude per joint')
        axes[1].legend(fontsize=8)
        fig.suptitle('Q3.1 – Per-joint neural metrics', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, 'ex3_1_per_joint_metrics.png'), dpi=200)
        plt.close()

    print("\nQ3.1 plots saved to:", PLOT_PATH)


# Q3.2 – w_ipsi sweep

def analysis_3_2():
    """
    Load sweep results from exercise3_2.py and plot CoT, forward speed,
    neural frequency, and neural amplitude vs w_ipsi.
    """
    print("\n" + "="*70)
    print("Q3.2 – w_ipsi sweep Analysis")
    print("="*70)

    # Discover available w_ipsi values from files present in BASE_PATH_32.
    if not os.path.isdir(BASE_PATH_32):
        print(f"  [SKIP] {BASE_PATH_32} not found. Run exercise3_2.py first.")
        return

    import glob, re

    hdf5_files = sorted(glob.glob(os.path.join(BASE_PATH_32, 'simulation_w_ipsi*.hdf5')))
    if not hdf5_files:
        print(f"  [SKIP] No simulation_w_ipsi*.hdf5 files found in {BASE_PATH_32}.")
        return

    # Parse w_ipsi values from filenames  (e.g. simulation_w_ipsi3.000.hdf5)
    pattern = re.compile(r'simulation_w_ipsi(-?[\d.]+)\.hdf5')
    w_vals_found = []
    for fp in hdf5_files:
        m = pattern.search(os.path.basename(fp))
        if m:
            w_vals_found.append(float(m.group(1)))
    if not w_vals_found:
        print(f"  [SKIP] Could not parse w_ipsi values from filenames.")
        return

    w_vals_found = sorted(w_vals_found)
    print(f"  Found {len(w_vals_found)} w_ipsi values: {w_vals_found}")

    results = {
        'w_ipsi':    [],
        'speed_fwd': [],
        'cot':       [],
        'freq_mean': [],
        'amp_mean':  [],
    }

    for w in w_vals_found:
        tag      = f'{float(w):0.3f}'
        hdf5_p   = os.path.join(BASE_PATH_32, f'simulation_w_ipsi{tag}.hdf5')
        pkl_p    = os.path.join(BASE_PATH_32, f'controller_w_ipsi{tag}.pkl')

        if not os.path.isfile(hdf5_p) or not os.path.isfile(pkl_p):
            print(f"  [SKIP] w_ipsi={w}: missing files.")
            continue

        (times, lp, lv, jp, jv, jt) = load_sim(hdf5_p)
        state, n_osc, n_body = load_controller(pkl_p)
        _, _, ml, mr         = unpack_controller(state, n_osc)
        metrics = compute_all_metrics(times, lp, lv, jp, jv, jt, ml, mr)

        results['w_ipsi'].append(w)
        results['speed_fwd'].append(metrics['speed_fwd'])
        results['cot'].append(metrics['cot'])
        results['freq_mean'].append(float(np.mean(metrics['freq'])))
        results['amp_mean'].append(float(np.mean(metrics['amp'])))

        print(f"  w_ipsi={w:6.2f}  speed={metrics['speed_fwd']:.4f} m/s"
              f"  CoT={metrics['cot']:.4f}  freq={np.mean(metrics['freq']):.3f} Hz"
              f"  amp={np.mean(metrics['amp']):.3f}")

    if not results['w_ipsi']:
        print("  No valid results – skipping Q3.2 plots.")
        return

    w_arr  = np.array(results['w_ipsi'])
    spd    = np.array(results['speed_fwd'])
    cot    = np.array(results['cot'])
    freq   = np.array(results['freq_mean'])
    amp    = np.array(results['amp_mean'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(w_arr, spd, 'o-', color='tab:blue')
    axes[0, 0].set_xlabel('w_ipsi')
    axes[0, 0].set_ylabel('Forward speed (m/s)')
    axes[0, 0].set_title('Forward speed vs w_ipsi')
    axes[0, 0].axvline(0, color='k', ls=':', lw=0.8, label='no feedback')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(w_arr, cot, 'o-', color='tab:red')
    axes[0, 1].set_xlabel('w_ipsi')
    axes[0, 1].set_ylabel('CoT (J/m)')
    axes[0, 1].set_title('Cost of Transport vs w_ipsi')
    axes[0, 1].axvline(0, color='k', ls=':', lw=0.8, label='no feedback')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(w_arr, freq, 'o-', color='tab:green')
    axes[1, 0].set_xlabel('w_ipsi')
    axes[1, 0].set_ylabel('Mean neural frequency (Hz)')
    axes[1, 0].set_title('Neural frequency vs w_ipsi')
    axes[1, 0].axvline(0, color='k', ls=':', lw=0.8, label='no feedback')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(w_arr, amp, 'o-', color='tab:purple')
    axes[1, 1].set_xlabel('w_ipsi')
    axes[1, 1].set_ylabel('Mean neural amplitude')
    axes[1, 1].set_title('Neural amplitude vs w_ipsi')
    axes[1, 1].axvline(0, color='k', ls=':', lw=0.8, label='no feedback')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Q3.2 – Effect of stretch feedback gain w_ipsi on locomotion metrics',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'ex3_2_wipsi_sweep.png'), dpi=200)
    plt.close()

    print("\nQ3.2 plot saved to:", PLOT_PATH)


# Combined summary table

def print_comparison_table():
    """Print a concise side-by-side metrics table for Q3.1."""
    print("\n" + "="*70)
    print("Q3.1 – Summary Table")
    print(f"{'Metric':<30} {'With SF':>15} {'Without SF':>15}")
    print("-"*60)

    def _load_metrics(hdf5_p, pkl_p):
        if not os.path.isfile(hdf5_p) or not os.path.isfile(pkl_p):
            return None
        (times, lp, lv, jp, jv, jt) = load_sim(hdf5_p)
        state, n_osc, _ = load_controller(pkl_p)
        _, _, ml, mr    = unpack_controller(state, n_osc)
        return compute_all_metrics(times, lp, lv, jp, jv, jt, ml, mr)

    m_with    = _load_metrics(
        BASE_PATH_31 + 'simulation_with_sf.hdf5',
        BASE_PATH_31 + 'controller_with_sf.pkl',
    )
    m_without = _load_metrics(
        BASE_PATH_31 + 'simulation_without_sf.hdf5',
        BASE_PATH_31 + 'controller_without_sf.pkl',
    )

    rows = [
        ('Forward speed (m/s)',     'speed_fwd',  False),
        ('Lateral speed (m/s)',     'speed_lat',  False),
        ('Energy (J)',              'energy',     False),
        ('CoT (J/m)',               'cot',        False),
        ('Mean IPL_neur (rad)',      'ipl_mean',   False),
        ('Mean neural freq (Hz)',   'freq',       True),
        ('Mean neural amp',         'amp',        True),
    ]
    for row_name, key, is_arr in rows:
        def _fmt(m):
            if m is None:
                return '   N/A'
            v = m[key]
            if is_arr:
                return f'{float(np.mean(v)):>15.4f}'
            return f'{float(v):>15.4f}'
        print(f"{row_name:<30} {_fmt(m_with)} {_fmt(m_without)}")
    print("="*70)


# Entry point

if __name__ == '__main__':
    analysis_3_1()
    print_comparison_table()
    analysis_3_2()
    print("\nAll done. Figures saved to:", PLOT_PATH)