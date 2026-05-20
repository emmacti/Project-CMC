"""[Project1] Script to call all exercises"""

from farms_core import pylog

from exercise_example import exercise_example
from exercise_p1 import exercise_1a_networks
from exercise_p2 import (
    exercise_walk,
    exercise_ramp_swim,
    exercise_ramp_walk,
)


def exercise_all(arguments):
    """Run all exercises"""

    verbose = 'not_verbose' not in arguments

    if not verbose:
        pylog.set_level('warning')

    # Timestep
    timestep = 5e-3
    if 'example' in arguments:
        exercise_example(timestep)
    if '1a' in arguments:
        exercise_1a_networks(plot=False, timestep=1e-2)  # don't show plot
    if '2a' in arguments:
        exercise_walk(timestep)
    if '2b' in arguments:
        exercise_ramp_swim(timestep)
    if '2c' in arguments:
        exercise_ramp_walk(timestep)

    if not verbose:
        pylog.set_level('debug')


if __name__ == '__main__':
    exercises = []
    exercises += ['1a']
    exercises += ['2a', '2b', '2c']
    exercise_all(arguments=exercises)

