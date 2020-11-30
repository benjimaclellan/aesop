"""
Test of topology optimization routines

TODO: HoF should be taken prior to speciation!!!!!
"""

# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform
import copy

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# various imports
import matplotlib.pyplot as plt
import psutil
import numpy as np
import config.config as config

from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evolver import ProbabilityLookupEvolver

from problems.example.node_types import TerminalSource, TerminalSink

from problems.example.evaluator_subclasses.evaluator_pulserep import PulseRepetition

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof

from lib.functions import parse_command_line_args

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path='pulse_rep_rate', unique_id=True)
    io.save_machine_metadata(io.save_path)

    ga_opts = {'n_generations': 8,
               'n_population': 8,
               'n_hof': 6,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()}

    propagator = Propagator(window_t=100e-9, n_samples=2**14, central_wl=1.55e-6)

    pulse_width, rep_t, peak_power = (0.3e-9, 20.0e-9, 1.0)
    p, q = (1, 2)

    input_laser = PulsedLaser(parameters_from_name={'pulse_width': pulse_width, 'peak_power': peak_power,
                                                    't_rep': rep_t, 'pulse_shape': 'gaussian',
                                                    'central_wl': 1.55e-6, 'train': True})
    input_laser.node_lock = True
    input_laser.protected = True

    input = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)
    target = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width * (p / q), rep_t=rep_t * (p / q), peak_power=peak_power * (p / q))
    evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)

    evolver = ProbabilityLookupEvolver(verbose=False)

    md = MeasurementDevice()
    md.protected = True

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             'sink': TerminalSink()}
    edges = {('source', 0): input_laser,
             (0, 'sink'): md,
             }
    graph = Graph.init_graph(nodes=nodes, edges=edges)
    update_rule = 'random'



    hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                     ga_opts=ga_opts, local_mode=False, update_rule='random',
                                     parameter_opt_method='L-BFGS+GA',
                                     include_dashboard=False, crossover_maker=None)

    save_hof(hof, io)
    plot_hof(hof, propagator, evaluator, io)

    fig, ax = plt.subplots(1, 1, figsize=[5, 3])
    ax.fill_between(log['generation'], log['best'], log['mean'], color='grey', alpha=0.2)
    ax.plot(log['generation'], log['best'], label='Best')
    ax.plot(log['generation'], log['mean'], label='Population mean')
    ax.plot(log['generation'], log['minimum'], color='darkgrey', label='Population minimum')
    ax.plot(log['generation'], log['maximum'], color='black', label='Population maximum')
    ax.set(xlabel='Generation', ylabel='Cost')
    ax.legend()

    io.save_fig(fig, 'topology_log.png')