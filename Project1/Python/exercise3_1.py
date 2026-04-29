#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

import os
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt

from farms_core import pylog
from farms_core.utils.profile import profile

from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    filter_signals,
    LINKS_MASSES,
)
from simulate import runsim


BASE_PATH = 'logs/exercise3_1/'
PLOT_PATH = 'results'

def _load_sim(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        times = f['times'][:]
        links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    links_pos = links[:, :, 7:10]
    links_vel = links[:, :, 14:17]
    joints_pos = joints[:, :, 0]
    joints_vel = joints[:, :, 1]
    joints_tau = joints[:, :, 2]
    return times, links_pos, links_vel, joints_pos, joints_vel, joints_tau


def _load_controller_state(pkl_path: str):
    with open(pkl_path, "rb") as f:
        controller_data = pickle.load(f)
    state = controller_data["state"]
    n_total = state.shape[1]
    if n_total % 3 != 0:
        raise ValueError(f"Unexpected controller state width={n_total}, expected multiple of 3.")
    n_osc = n_total // 3
    phases = state[:, :n_osc]
    amps = state[:, n_osc:2*n_osc]
    motor_storage = state[:, 2*n_osc:3*n_osc]
    motor_left = motor_storage[:, 0::2]
    motor_right = motor_storage[:, 1::2]
    neural = motor_left - motor_right
    return phases, amps, motor_left, motor_right, neural


def _metrics_for_case(hdf5_path: str, controller_pkl: str):
    times, links_pos, links_vel, jp, jv, jt = _load_sim(hdf5_path)
    phases, amps, ml, mr, neural = _load_controller_state(controller_pkl)

    neural_s = filter_signals(times=times, signals=neural)
    freqs, _, amps_neur = compute_frequency_amplitude_fft(times=times, smooth_signals=neural_s)

    speed_fwd, _speed_lat = compute_mechanical_speed(links_positions=links_pos, links_velocities=links_vel)
    _energy, cot = compute_mechanical_energy_and_cot(
        times=times,
        links_positions=links_pos,
        joints_torques=jt,
        joints_velocities=jv,
    )
    return {
        "times": times,
        "links_pos": links_pos,
        "joints_pos": jp,
        "phases": phases,
        "amps": amps,
        "ml": ml,
        "mr": mr,
        "neural": neural,
        "freqs": freqs,
        "amps_neur": amps_neur,
        "speed_fwd": float(speed_fwd),
        "cot": float(cot),
    }


def _plot_case(prefix: str, m: dict, max_seconds: float = 5.0, n_joints_plot: int = 3):
    os.makedirs(PLOT_PATH, exist_ok=True)
    t = m["times"]
    mask = t <= (t[0] + max_seconds)

    phases = m["phases"]
    amps = m["amps"]
    ml = m["ml"]
    mr = m["mr"]
    neural = m["neural"]

    # Plot theta and r for first joints (interleaved oscillators)
    plt.figure(figsize=(9, 4))
    for j in range(n_joints_plot):
        plt.plot(t[mask], phases[mask, 2*j], label=f"theta L joint {j}")
        plt.plot(t[mask], phases[mask, 2*j+1], ls="--", label=f"theta R joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("theta (rad)")
    plt.title(f"{prefix}: oscillator phases (first {max_seconds:.0f}s)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"{prefix}_theta.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4))
    for j in range(n_joints_plot):
        plt.plot(t[mask], amps[mask, 2*j], label=f"r L joint {j}")
        plt.plot(t[mask], amps[mask, 2*j+1], ls="--", label=f"r R joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("r")
    plt.title(f"{prefix}: oscillator amplitudes (first {max_seconds:.0f}s)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"{prefix}_r.png"), dpi=200)
    plt.close()

    # Sum/diff muscle outputs from motor storage
    m_sum = ml + mr

    plt.figure(figsize=(9, 4))
    for j in range(n_joints_plot):
        plt.plot(t[mask], m_sum[mask, j], label=f"ML+MR joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("ML+MR")
    plt.title(f"{prefix}: muscle output sum (first {max_seconds:.0f}s)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"{prefix}_muscle_sum.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4))
    for j in range(n_joints_plot):
        plt.plot(t[mask], neural[mask, j], label=f"ML-MR joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("ML-MR")
    plt.title(f"{prefix}: muscle output difference (first {max_seconds:.0f}s)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"{prefix}_muscle_diff.png"), dpi=200)
    plt.close()


def _plot_comparison(with_sf: dict, without_sf: dict, max_seconds: float = 5.0, n_joints_plot: int = 3):
    os.makedirs(PLOT_PATH, exist_ok=True)

    def _mask(m):
        t = m["times"]
        return t, (t <= (t[0] + max_seconds))

    def _two_panel(fname: str, y_label: str, title: str, plot_fn):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)
        for ax, m, case_title in [
            (axes[0], with_sf, "with stretch feedback"),
            (axes[1], without_sf, "without stretch feedback"),
        ]:
            t, mask = _mask(m)
            plot_fn(ax, t, mask, m)
            ax.set_title(case_title, fontsize=9)
            ax.set_xlabel("time (s)")
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel(y_label)
        fig.suptitle(f"{title} (first {max_seconds:.0f}s)", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, fname), dpi=200)
        plt.close()

    def _plot_theta(ax, t, mask, m):
        phases = m["phases"]
        for j in range(n_joints_plot):
            ax.plot(t[mask], phases[mask, 2 * j], label=f"theta L joint {j}")
            ax.plot(t[mask], phases[mask, 2 * j + 1], ls="--", label=f"theta R joint {j}")
        ax.legend(fontsize=8, ncol=2)

    def _plot_r(ax, t, mask, m):
        amps = m["amps"]
        for j in range(n_joints_plot):
            ax.plot(t[mask], amps[mask, 2 * j], label=f"r L joint {j}")
            ax.plot(t[mask], amps[mask, 2 * j + 1], ls="--", label=f"r R joint {j}")
        ax.legend(fontsize=8, ncol=2)

    def _plot_muscle_sum(ax, t, mask, m):
        m_sum = m["ml"] + m["mr"]
        for j in range(n_joints_plot):
            ax.plot(t[mask], m_sum[mask, j], label=f"ML+MR joint {j}")
        ax.legend(fontsize=8, ncol=2)

    def _plot_muscle_diff(ax, t, mask, m):
        neural = m["neural"]
        for j in range(n_joints_plot):
            ax.plot(t[mask], neural[mask, j], label=f"ML-MR joint {j}")
        ax.legend(fontsize=8, ncol=2)

    _two_panel(
        fname="ex3_1_theta.png",
        y_label="theta (rad)",
        title="Q3.1 – Oscillator phases",
        plot_fn=_plot_theta,
    )
    _two_panel(
        fname="ex3_1_r.png",
        y_label="r",
        title="Q3.1 – Oscillator amplitudes",
        plot_fn=_plot_r,
    )
    _two_panel(
        fname="ex3_1_muscle_sum.png",
        y_label="ML+MR",
        title="Q3.1 – Muscle output sum",
        plot_fn=_plot_muscle_sum,
    )
    _two_panel(
        fname="ex3_1_muscle_diff.png",
        y_label="ML-MR",
        title="Q3.1 – Muscle output difference",
        plot_fn=_plot_muscle_diff,
    )


def _com_xy(links_positions: np.ndarray) -> np.ndarray:
    n_links_total = links_positions.shape[1]
    masses = LINKS_MASSES[:n_links_total]
    mass_sum = float(masses.sum())
    com_pos = (links_positions * masses[None, :, None]).sum(axis=1) / mass_sum
    return com_pos[:, :2]


def _plot_q31_missing_figures(with_sf: dict, without_sf: dict, max_seconds: float = 5.0):
    """
    Recreate older Q3.1 figures that existed in `old figure/`:
      - ex3_1_joint_angles.png
      - ex3_1_com_trajectory.png
      - ex3_1_metric_comparison.png
      - ex3_1_per_joint_metrics.png
    """
    os.makedirs(PLOT_PATH, exist_ok=True)

    # Joint angles (first 8 joints), 2-panel
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True, sharex=True)
    for ax, m, case_title in [
        (axes[0], with_sf, "With SF (w_ipsi=3.0)"),
        (axes[1], without_sf, "Without SF (w_ipsi=0)"),
    ]:
        t = m["times"]
        mask = t <= (t[0] + max_seconds)
        jp = m["joints_pos"]
        n_plot = min(8, jp.shape[1])
        for j in range(n_plot):
            ax.plot(t[mask], jp[mask, j], lw=1.0, label=f"joint {j}")
        ax.set_title(case_title, fontsize=10)
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=7)
    axes[0].set_ylabel("joint angle (rad)")
    fig.suptitle("Q3.1 – Joint angles (first 8 joints)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "ex3_1_joint_angles.png"), dpi=200)
    plt.close()

    # CoM trajectory (XY), 2-panel, equal aspect
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m, case_title in [
        (axes[0], with_sf, "With SF (w_ipsi=3.0)"),
        (axes[1], without_sf, "Without SF (w_ipsi=0)"),
    ]:
        xy = _com_xy(m["links_pos"])
        ax.plot(xy[:, 0], xy[:, 1], lw=2.0)
        ax.set_title(case_title, fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.15)
    fig.suptitle("Q3.1 – CoM trajectory (XY)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "ex3_1_com_trajectory.png"), dpi=200)
    plt.close()

    # Metric comparison: speed, CoT, mean freq, mean amp
    def _mean(x):
        x = np.asarray(x)
        return float(np.mean(x))

    with_vals = [
        float(with_sf["speed_fwd"]),
        float(with_sf["cot"]),
        _mean(with_sf["freqs"]),
        _mean(with_sf["amps_neur"]),
    ]
    without_vals = [
        float(without_sf["speed_fwd"]),
        float(without_sf["cot"]),
        _mean(without_sf["freqs"]),
        _mean(without_sf["amps_neur"]),
    ]
    titles = ["Forward speed (m/s)", "CoT (J/m)", "Mean neural freq (Hz)", "Mean neural amp"]
    ylabels = ["Forward speed (m/s)", "CoT (J/m)", "Mean neural freq (Hz)", "Mean neural amp"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    fig.suptitle("Q3.1 – Metric comparison: with vs without stretch feedback", fontsize=11)
    xlabels = ["With SF (w_ipsi=3.0)", "Without SF (w_ipsi=0)"]
    for ax, title, ylabel, a, b in zip(axes, titles, ylabels, with_vals, without_vals):
        bars = ax.bar([0, 1], [a, b], color=["tab:blue", "tab:orange"], width=0.55)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks([0, 1], xlabels, rotation=25, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        for r in bars:
            ax.text(
                r.get_x() + r.get_width() / 2,
                r.get_height(),
                f"{r.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "ex3_1_metric_comparison.png"), dpi=200)
    plt.close()

    # Per-joint metrics: frequency and amplitude per joint (indices 0..7)
    def _per_joint(x):
        x = np.asarray(x)
        if x.ndim == 0:
            return np.full(8, float(x))
        if x.ndim == 1:
            return x[:8]
        # If time-varying per joint, average over time axis
        return np.mean(x, axis=0)[:8]

    f_with = _per_joint(with_sf["freqs"])
    f_without = _per_joint(without_sf["freqs"])
    a_with = _per_joint(with_sf["amps_neur"])
    a_without = _per_joint(without_sf["amps_neur"])
    j = np.arange(len(f_with))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Q3.1 – Per-joint neural metrics", fontsize=12)

    ax = axes[0]
    ax.plot(j, f_with, "o-", label="With SF (w_ipsi=3.0)")
    ax.plot(j, f_without, "o--", label="Without SF (w_ipsi=0)")
    ax.set_title("Neural frequency per joint", fontsize=11)
    ax.set_xlabel("joint index")
    ax.set_ylabel("Neural frequency (Hz)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(j, a_with, "o-", label="With SF (w_ipsi=3.0)")
    ax.plot(j, a_without, "o--", label="Without SF (w_ipsi=0)")
    ax.set_title("Neural amplitude per joint", fontsize=11)
    ax.set_xlabel("joint index")
    ax.set_ylabel("Neural amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "ex3_1_per_joint_metrics.png"), dpi=200)
    plt.close()


def main(**kwargs):
    """
    Q3.1 – Stretch feedback ablation (with vs without sensory feedback).

    Runs:
    - with stretch feedback: w_ipsi = 3.0
    - without stretch feedback: w_ipsi = 0.0

    Then post-processes the two runs to produce the plots/metrics requested in Q3.1.
    """
    controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': 3,
            'drive_right': 3,
            'd_low': 1,
            'd_high': 5,
            'a_rate': np.ones(8) * 3,
            'offset_freq': np.ones(8) * 1,
            'offset_amp': np.ones(8) * 0.5,
            'G_freq': np.ones(8) * 0.5,
            'G_amp': np.ones(8) * 0.25,
            'PL': np.ones(7) * np.pi * 2 / 8,
            'coupling_weights_rostral': 5,
            'coupling_weights_caudal': 5,
            'coupling_weights_contra': 10,
            'init_phase': np.random.default_rng(
                seed=42).uniform(
                0.0,
                2 * np.pi,
                size=16),
        },
    }
    w_ipsi = 3.0
    fast = kwargs.pop('fast', False)
    headless = kwargs.pop('headless', False)

    
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=w_ipsi,
        recording='animation3_1_with_sf.mp4',
        hdf5_name='simulation_with_sf.hdf5',
        controller_name='controller_with_sf.pkl',
        runtime_n_iterations=5001,
        runtime_buffer_size=5001,
        fast=fast,
        headless=headless,
    )
    
    print("START without sensory feedback")
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=0,
        recording='animation3_1_without_sf.mp4',
        hdf5_name='simulation_without_sf.hdf5',
        controller_name='controller_without_sf.pkl',
        runtime_n_iterations=5001,
        runtime_buffer_size=5001,
        fast=fast,
        headless=headless,
    )
    print("DONE")

    # Post-process: 
    os.makedirs(PLOT_PATH, exist_ok=True)
    with_sf = _metrics_for_case(
        hdf5_path=os.path.join(BASE_PATH, "simulation_with_sf.hdf5"),
        controller_pkl=os.path.join(BASE_PATH, "controller_with_sf.pkl"),
    )
    without_sf = _metrics_for_case(
        hdf5_path=os.path.join(BASE_PATH, "simulation_without_sf.hdf5"),
        controller_pkl=os.path.join(BASE_PATH, "controller_without_sf.pkl"),
    )

    _plot_comparison(with_sf=with_sf, without_sf=without_sf)
    _plot_q31_missing_figures(with_sf=with_sf, without_sf=without_sf)

    print("\nMetrics comparison (Q3.1)")
    print(f"  with SF   : speed_fwd={with_sf['speed_fwd']:.4f} m/s, CoT={with_sf['cot']:.4f}")
    print(f"  without SF: speed_fwd={without_sf['speed_fwd']:.4f} m/s, CoT={without_sf['cot']:.4f}")

def exercise3_1(**kwargs):
    """ex3.1 main"""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)
    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise3_1(plot=True)

