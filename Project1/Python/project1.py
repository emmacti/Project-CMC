"""Project 1"""

from multiprocessing import Pool
from farms_core import pylog
from exercise_all import exercise_all


def main(parallel=False):
    """Main function that runs all the exercises."""

    pylog.info('Running simulation exercises')

    arguments = (
        []
    )

    arguments = (
        []
    )

    if parallel:
        with Pool() as pool:  # Pool(processes=4)
            pool.map(exercise_all, [[arg] for arg in arguments])
    else:
        exercise_all(arguments=arguments)

    pylog.info('Plotting simulation results')


if __name__ == '__main__':
    main()

