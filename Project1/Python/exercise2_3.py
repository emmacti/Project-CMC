#!/usr/bin/env python3

import os
import h5py
import matplotlib
matplotlib.use('Agg') # I added this
import matplotlib.pyplot as plt


from farms_core import pylog

from cmc_controllers.metrics import *
from simulate import runsim


BASE_PATH = 'logs/exercise2_3/'
PLOT_PATH = 'results'
ANIMAL_DATA_PATH = 'cmc_project_pack/models/a2sw5_cycle_smoothed.csv'

"""
def get_animal_data(path):
    #Load animal data
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    freq = np.zeros(8)
    amp = np.zeros(8)
    ipls = np.zeros(7)
    times = data[10:-10, 0]
    joint_angles = data[10:-10, 1:9]
    return freq, np.deg2rad(amp), ipls
"""
def get_animal_data(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)

    times = data[10:-10, 0]
    joint_angles = data[10:-10, 1:9]

    # Smooth signals
    smooth = filter_signals(times=times, signals=joint_angles)

    # Frequency & amplitude
    freqs, _, amps = compute_frequency_amplitude_fft(
        times=times,
        smooth_signals=smooth,
    )

    # IPL (between adjacent joints)
    inds = [[i, i+1] for i in range(7)]
    _, ipls = compute_neural_phase_lags(
        times=times,
        smooth_signals=smooth,
        freqs=freqs,
        inds_couples=inds,
    )

    return freqs, amps, ipls

def exercise2_3(**kwargs):
    """ex2.3 main"""
    pylog.warning("TODO: 2.3 Analyze the provided animal data and compare the animal locomotion performance with your implemented controller.")
    # pylog.set_level('critical')

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise2_3(plot=True)

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
        'PL': float(2*np.pi/8),
        'coupling_weights_rostral': 5,
        'coupling_weights_caudal': 5,
        'coupling_weights_contra': 10,
        'init_phase': np.ascontiguousarray(
            np.random.uniform(0, 2*np.pi, 16)
        )
    }
}

runsim(
    controller=controller,
    base_path=BASE_PATH,
    headless=True,
    fast=True,
)

with h5py.File(BASE_PATH + "simulation.hdf5", "r") as f:
    times = f['times'][:]
    joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:, :, 0]
    


smooth = filter_signals(times=times, signals=joints)

freq_ctrl, _, amp_ctrl = compute_frequency_amplitude_fft(
    times=times,
    smooth_signals=smooth,
)

inds = [[i, i+1] for i in range(7)]
_, ipls_ctrl = compute_neural_phase_lags(
    times=times,
    smooth_signals=smooth,
    freqs=freq_ctrl,
    inds_couples=inds,
)

freq_animal, amp_animal, ipls_animal = get_animal_data(ANIMAL_DATA_PATH)

plt.figure()
plt.plot(freq_animal, 'o-', label='animal')
plt.plot(freq_ctrl, 's--', label='controller')
plt.title("Frequency comparison")
plt.xlabel("Joint index")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.savefig("results/freq.png")

plt.figure()
plt.plot(amp_animal, 'o-', label='animal')
plt.plot(amp_ctrl, 's--', label='controller')
plt.title("Amplitude comparison")
plt.xlabel("Joint index")
plt.ylabel("Amplitude (rad)")
plt.legend()
plt.savefig("results/amp.png")


plt.figure()
plt.plot(ipls_animal, 'o-', label='animal')
plt.plot(ipls_ctrl, 's--', label='controller')
plt.title("IPL comparison")
plt.xlabel("Segment")
plt.ylabel("Phase lag (rad)")
plt.legend()
plt.savefig("results/ipl.png")

print("freq_ctrl:", freq_ctrl)
print("amp_ctrl:", amp_ctrl)
print("ipls_ctrl:", ipls_ctrl)