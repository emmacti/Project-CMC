"""[Project1] Script to call all exercises"""


from farms_core import pylog
from exercise1_1 import exercise1_1
from exercise1_2 import exercise1_2
from exercise2_1 import exercise2_1
from exercise2_2 import exercise2_2
from exercise2_3 import exercise2_3
from exercise3_1 import exercise3_1
from exercise3_2 import exercise3_2
from exercise3_3 import exercise3_3

show_plot = False


def exercise_all(arguments):
    """Run all exercises"""

    verbose = 'not_verbose' not in arguments

    if not verbose:
        pylog.set_level('warning')

    if '1_1' in arguments:
        exercise1_1(plot=show_plot, fast=True, headless=True)

    if '1_2' in arguments:
        exercise1_2(plot=show_plot)

    if '2_1' in arguments:
        exercise2_1(plot=show_plot, fast=True, headless=True)

    if '2_2' in arguments:
        exercise2_2(plot=show_plot)

    if '2_3' in arguments:
        exercise2_3(plot=show_plot)

    if '3_1' in arguments:
        exercise3_1(plot=show_plot, fast=True, headless=True)

    if '3_2' in arguments:
        exercise3_2(plot=show_plot)

    if '3_3' in arguments:
        exercise3_3(plot=show_plot)

