
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.extensions.extensions import import_item
from farms_sim.utils.parse_args import sim_parse_args
from farms_sim.simulation import (
    setup_from_clargs,
    run_simulation,
)
import pickle
import itertools
from cmc_controllers.metrics import *


def _as_filename_token(value):
    """Convert parameter values into filesystem-safe short tokens."""
    # Numpy scalar-like values can expose shape/flat and recurse forever.
    if hasattr(
            value,
            'shape') and getattr(
            value,
            'shape',
            None) == () and hasattr(
                value,
            'item'):
        value = value.item()

    if isinstance(value, (list, tuple)):
        if not value:
            return 'empty'
        return f"seq{len(value)}_{_as_filename_token(value[0])}"

    # Support numpy arrays without importing numpy explicitly here.
    if hasattr(value, 'shape') and hasattr(value, 'flat'):
        size = getattr(value, 'size', 0)
        if size == 0:
            return 'empty'
        first = value.flat[0]
        if first is value:
            return f"arr{size}"
        return f"arr{size}_{_as_filename_token(first)}"

    if isinstance(value, (int, float)):
        return f"{float(value):0.3f}"

    text = str(value)
    safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in text)
    return safe[:40]


def _build_default_output_names(params):
    """Build default log filenames from a parameter dictionary."""
    suffix = '_'.join(
        f"{key}{_as_filename_token(val)}" for key, val in params.items()
    )
    return (
        f"simulation_{suffix}.hdf5",
        f"controller_{suffix}.pkl",
    )


def build_parameter_space(parameter_grid):
    """Build a list of parameter dictionaries from a named grid."""
    parameter_names = list(parameter_grid.keys())
    return [
        dict(zip(parameter_names, values))
        for values in itertools.product(
            *(parameter_grid[name] for name in parameter_names)
        )
    ]


def runsim(
    controller,
    base_path,
    **kwargs
):
    """Main Simulation function"""
    os.makedirs(base_path, exist_ok=True)
    headless = kwargs.pop('headless', False)
    fast = kwargs.pop('fast', False)
    recording = kwargs.pop('recording', None)
    hdf5_name = kwargs.pop('hdf5_name', 'simulation.hdf5')
    controller_name = kwargs.pop('controller_name', 'controller.pkl')
    runtime_n_iterations = kwargs.pop('runtime_n_iterations', None)
    runtime_buffer_size = kwargs.pop('runtime_buffer_size', None)

    # Setup
    pylog.info('Loading options from clargs')
    clargs = sim_parse_args()

    clargs.experiment_config = os.path.join(
        os.path.dirname(__file__),
        "cmc_project_pack",
        "configs",
        "experiment_config.yaml")
    _, exp_options, simulator = setup_from_clargs(clargs=clargs)

    # disable visulization for faster simulation
    exp_options.simulation.runtime.headless = headless
    exp_options.simulation.runtime.fast = fast
    if runtime_n_iterations is not None:
        exp_options.simulation.runtime.n_iterations = runtime_n_iterations
    if runtime_buffer_size is not None:
        exp_options.simulation.runtime.buffer_size = runtime_buffer_size

    controller.setdefault('config', {})
    param_aliases = {
        'phaselag': 'PL',
    }
    for key, value in kwargs.items():
        if key == 'drive':
            controller['config']['drive_left'] = value
            controller['config']['drive_right'] = value
            continue
        target_key = param_aliases.get(key, key)
        controller['config'][target_key] = value
    exp_options.animats[0].extensions.insert(0, controller)

    # Logger
    for ext in exp_options.simulation.extensions:
        if ext['loader'] == 'farms_core.simulation.extensions.ExperimentLogger':
            ext['config']['log_path'] = base_path
            ext['config']['log_name'] = hdf5_name
        if ext['loader'] == 'farms_mujoco.sensors.camera.CameraRecording':
            if recording:
                ext['config']['path'] = base_path + recording
            else:
                exp_options.simulation.extensions.remove(ext)
    for ext in exp_options.simulation.extensions:
        if ext['loader'] == 'farms_core.simulation.extensions.ExperimentLogger':
            ext['config']['log_path'] = base_path
            ext['config']['log_name'] = hdf5_name
        if ext['loader'] == 'farms_mujoco.sensors.camera.CameraRecording':
            ext['config']['path'] = base_path + recording
            if not recording:
                exp_options.simulation.extensions.remove(ext)
    # Data
    experiment_data_loader = import_item(exp_options.loaders.experiment_data)
    experiment_data = experiment_data_loader.from_options(exp_options)

    # Simulation
    pylog.info('Creating simulation environment')
    sim = run_simulation(
        experiment_data=experiment_data,
        experiment_options=exp_options,
        simulator=simulator,
    )

    # Dump controller
    pylog.info('Saving controller')
    controller = sim.task._controllers[0]
    controller_state = controller.network.state
    indices = {
        'left_body_idx': controller.left_body_idx,
        'right_body_idx': controller.right_body_idx,
    }
    controller_data = {
        "state": controller_state,
        "indices": indices, }
    with open(base_path + controller_name, "wb") as f:
        pickle.dump(controller_data, f)


def run_multiple(
    max_workers,
    controller,
    base_path,
    parameter_grid,
    common_kwargs=None,
):
    parameter_space = build_parameter_space(parameter_grid)
    common_kwargs = {} if common_kwargs is None else dict(common_kwargs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for params in parameter_space:
            if not isinstance(params, dict):
                raise TypeError(
                    'Each entry in parameter_space must be a dict of kwargs '
                    'for runsim().'
                )

            run_kwargs = dict(common_kwargs)
            run_kwargs.update(params)

            if 'hdf5_name' not in run_kwargs or 'controller_name' not in run_kwargs:
                hdf5_name, controller_name = _build_default_output_names(
                    params)
                run_kwargs.setdefault('hdf5_name', hdf5_name)
                run_kwargs.setdefault('controller_name', controller_name)

            future = executor.submit(
                runsim,
                controller=controller,
                base_path=base_path,
                **run_kwargs,
            )
            futures[future] = params

        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Simulations') if tqdm else as_completed(futures)
        for future in progress:
            try:
                future.result()
            except Exception as e:
                print(f"Simulation failed for params={futures[future]}: {e}")

