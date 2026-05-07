"""Save figures"""
import os
import matplotlib.pyplot as plt
from farms_core import pylog


def save_figure(figure, dir='results', name=None, **kwargs):
    """ Save figure """
    extensions = kwargs.pop('extensions', ['pdf'])
    fig_obj = plt.figure(figure)
    fname = figure.replace(' ', '_').replace('.', 'dot')
    for extension in extensions:
        path = f'{dir}/{fname}.{extension}' if name is None else f'{dir}/{fname}.{extension}'
        fig_obj.savefig(path, bbox_inches='tight')
        pylog.debug('Saving figure %s...', path)


def save_figures(**kwargs):
    """Save_figures"""
    figures = [str(figure) for figure in plt.get_figlabels()]
    pylog.debug('Other files:\n    - %s', '\n    - '.join(figures))
    os.makedirs('./results/', exist_ok=True)
    extensions = kwargs.pop('extensions', ['pdf'])
    for name in figures:
        save_figure(name, extensions=extensions)

