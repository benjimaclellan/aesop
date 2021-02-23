"""
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

from lib.functions import InputOutput

from lib.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.evolver import HessianProbabilityEvolver, OperatorBasedProbEvolver

from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import Photodiode
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types_subclasses.inputs import PulsedLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import PhaseModulator

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof
from lib.functions import parse_command_line_args


plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path='rfawg', unique_id=False)
    io.save_machine_metadata(io.save_path)

    ga_opts = {'n_generations': 8,
               'n_population': 8,
               'n_hof': 6,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=2e-9, n_samples=2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9,
                                                 target_amplitude=0.02, target_waveform='saw',)
    # evolver = HessianProbabilityEvolver(verbose=False)
    evolver = OperatorBasedProbEvolver(verbose=False)

    pd = Photodiode()
    pd.protected = True

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             'sink': TerminalSink()}
    edges = {('source', 0): ContinuousWaveLaser(),
             (0, 'sink'): pd,
             }
    graph = Graph.init_graph(nodes=nodes, edges=edges)

    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator)

    update_rule = 'tournament'
    hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                     ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                     parameter_opt_method='L-BFGS+GA',
                                     include_dashboard=False, crossover_maker=None)

    save_hof(hof, io)
    plot_hof(hof, propagator, evaluator, io)

    fig, ax = plt.subplots(1, 1, figsize=[5,3])
    ax.fill_between(log['generation'], log['best'], log['mean'], color='grey', alpha=0.2)
    ax.plot(log['generation'], log['best'], label='Best')
    ax.plot(log['generation'], log['mean'], label='Population mean')
    ax.plot(log['generation'], log['minimum'], color='darkgrey', label='Population minimum')
    ax.plot(log['generation'], log['maximum'], color='black', label='Population maximum')
    ax.set(xlabel='Generation', ylabel='Cost')
    ax.legend()

    io.save_fig(fig, 'topology_log.png')
