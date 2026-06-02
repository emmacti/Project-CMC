"""[Project2] Script to call exercises"""

from farms_core import pylog

from exercise_example import exercise_example
from exercise_p1 import exercise_1a_networks
from exercise_p2 import (
    exercise_walk,
    exercise_ramp_swim,
    exercise_ramp_walk,
)
from exercise_p3 import (
    exercise_3_disable_limb_spine_coupling,
    exercise_3_limb_spine_antiphase,
    exercise_3a_coordination,
    exercise_3b_coordination,
)
from exercise_p4 import exercise_4a_transition


def exercise_all(arguments):
    """Run all exercises"""

    verbose = 'not_verbose' not in arguments

    if not verbose:
        pylog.set_level('warning')

    # Timestep
    timestep = 5e-3
    if 'example' in arguments:
        print('Running Exercise: example', flush=True)
        exercise_example(timestep)
    if '1a' in arguments:
        print('Running Exercise: 1a (Network & Dynamics)', flush=True)
        exercise_1a_networks(plot=False, timestep=1e-2)
    if '2a' in arguments:
        print('Running Exercise: 2a (Walking video)', flush=True)
        exercise_walk(timestep)
    if '2b' in arguments:
        print('Running Exercise: 2b (Ramp swim video)', flush=True)
        exercise_ramp_swim(timestep)
    if '2c' in arguments:
        print('Running Exercise: 2c (Ramp walk video)', flush=True)
        exercise_ramp_walk(timestep)

    # Project 2 - Part B
    if '3a' in arguments:
        print('Running Exercise: 3a (Disable limb-spine coupling)', flush=True)
        exercise_3_disable_limb_spine_coupling(timestep)
    if '3b' in arguments:
        print('Running Exercise: 3b (Limb-spine anti-phase)', flush=True)
        exercise_3_limb_spine_antiphase(timestep)
    if '3c' in arguments:
        print('Running Exercise: 3c (2D sweep: limb-spine phase offset vs drive)', flush=True)
        exercise_3a_coordination(timestep)
    if '3d' in arguments:
        print('Running Exercise: 3d (2D sweep: axial & limb amplitudes)', flush=True)
        exercise_3b_coordination(timestep)
    if '4a' in arguments:
        print('Running Exercise: 4a (Walk<->Swim transitions)', flush=True)
        exercise_4a_transition(timestep)

    if not verbose:
        pylog.set_level('debug')


if __name__ == '__main__':
    exercises = []
    exercises += ['example']
    exercises += ['1a']
    exercises += ['2a', '2b', '2c']
    exercises += ['3a', '3b', '3c', '3d']
    exercises += ['4a']
    exercise_all(arguments=exercises)

