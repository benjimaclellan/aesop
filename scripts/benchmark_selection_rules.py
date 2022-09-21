"""
Benchmark the different selection rules:
1. Random
2. Roulette
3. Tournament

With elitism set to 0.1, on a population of 10
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
import random
import autograd.numpy as np

from simulator.fiber.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.evolver import ProbabilityLookupEvolver

from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from simulator.fiber.node_types_subclasses.inputs import ContinuousWaveLaser
from simulator.fiber.node_types_subclasses.outputs import Photodiode
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof

from lib.functions import parse_command_line_args
from lib.functions import InputOutput

def benchmark_selection_rules(selection_rules, start_graph, evaluator, ga_opts, io):
    top_level_path = io.path
    repetitions = 10

    for selection_rule in selection_rules:
        io.path = top_level_path
        io.init_save_dir(sub_path=f"selectionRule_{selection_rule}", unique_id=False)
        io.path = io.save_path
    
        for i in range(repetitions):
            random.seed(i)
            np.random.seed(i)
            io.init_save_dir(sub_path=f"iteration_{i}", unique_id=False)

            hof, log = topology_optimization(copy.deepcopy(start_graph), propagator, evaluator, evolver, io,
                                             ga_opts=ga_opts, local_mode=False, update_rule=selection_rule,
                                             parameter_opt_method='L-BFGS+GA', elitism_ratio=0.1,
                                             include_dashboard=False, crossover_maker=None)

            save_hof(hof, io)
            plot_hof(hof, propagator, evaluator, io)

            fig, ax = plt.subplots(1, 1, figsize=[5,3])
            ax.fill_between(log['generation'].to_numpy(dtype='float'), log['best'].to_numpy(dtype='float'), log['mean'].to_numpy(dtype='float'), color='grey', alpha=0.2)
            ax.plot(log['generation'], log['best'], label='Best')
            ax.plot(log['generation'], log['mean'], label='Population mean')
            ax.plot(log['generation'], log['minimum'], color='darkgrey', label='Population minimum')
            ax.plot(log['generation'], log['maximum'], color='black', label='Population maximum')
            ax.set(xlabel='Generation', ylabel='Cost')
            ax.legend()

            io.save_fig(fig, 'topology_log.png')
            plt.close()

def rfawg_start_graph():
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
    return graph

if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=f'benchmark_selection_rules_{options_cl.evaluator}', unique_id=True)
    io.save_machine_metadata(io.save_path)
    io.path = io.save_path
    ga_opts = {'n_generations': 16,
               'n_hof': 6,
               'n_population': 16,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=1e-9, n_samples=2**14, central_wl=1.55e-6)

    if options_cl.evaluator == 'rfawg_saw':
        evaluator = RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9,
                                                    target_amplitude=0.02, target_waveform='saw')
        start_graph = rfawg_start_graph()
    elif options_cl.evaluator == 'rfawg_square':
        evaluator = RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9,
                                                    target_amplitude=0.02, target_waveform='square')   
        start_graph = rfawg_start_graph()
    else:
        raise ValueError(f'evaluator option {options_cl.evaluator} not valid. Options are: rfawg_saw, rfawg_square)')

    evolver = ProbabilityLookupEvolver(verbose=False)
    benchmark_selection_rules(['random', 'roulette', 'tournament'], start_graph, evaluator, ga_opts, io)
