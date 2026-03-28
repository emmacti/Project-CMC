#!/usr/bin/env python3

import os
import h5py
import matplotlib.pyplot as plt

from farms_core import pylog

from cmc_controllers.metrics import *
from simulate import runsim


BASE_PATH = 'logs/exercise2_3/'
PLOT_PATH = 'results'
ANIMAL_DATA_PATH = 'cmc_project_pack/models/a2sw5_cycle_smoothed.csv'


def get_animal_data(path):
    """Load animal data"""
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    freq = np.zeros(8)
    amp = np.zeros(8)
    ipls = np.zeros(7)
    times = data[10:-10, 0]
    joint_angles = data[10:-10, 1:9]
    return freq, np.deg2rad(amp), ipls


def exercise2_3(**kwargs):
    """ex2.3 main"""
    pylog.warning("TODO: 2.3 Analyze the provided animal data and compare the animal locomotion performance with your implemented controller.")
    # pylog.set_level('critical')

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise2_3(plot=True)

