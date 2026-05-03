"""Save figures"""
import os
import matplotlib.pyplot as plt
from farms_core import pylog


def save_figure(figure, dir='results', name=None, **kwargs):
    """ Save figure """
    for extension in kwargs.pop('extensions', ['pdf']):
        fig = figure.replace(' ', '_').replace('.', 'dot')
        if name is None:
            name = f'{dir}/{fig}.{extension}'
        else:
            name = f'{dir}/{fig}.{extension}'
        fig = plt.figure(figure)
        size = plt.rcParams.get('figure.figsize')
        fig.set_size_inches(0.7*size[0], 0.7*size[1], forward=True)
        fig.savefig(name, bbox_inches='tight')
        pylog.debug('Saving figure %s...', name)
        fig.set_size_inches(size[0], size[1], forward=True)


def save_figures(**kwargs):
    """Save_figures"""
    figures = [str(figure) for figure in plt.get_figlabels()]
    pylog.debug('Other files:\n    - %s', '\n    - '.join(figures))
    os.makedirs('./results/', exist_ok=True)
    for name in figures:
        save_figure(name, extensions=kwargs.pop('extensions', ['pdf']))

