"""
Test of topology optimization routines
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

import config.config as config

from lib.functions import InputOutput

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_
from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.evolver import ProbabilityLookupEvolver, OperatorBasedProbEvolver
from problems.example.evolution_operators.evolution_operators import AddSeriesComponent, RemoveComponent, SwapComponent, AddParallelComponent

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types import TerminalSource, TerminalSink

from algorithms.topology_optimization import topology_optimization

def handle_io():
    io = InputOutput(directory='testing', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=True)
    io.save_machine_metadata(io.save_path)
    return io

plt.close('all')
if __name__ == '__main__':
    ga_opts = {'n_generations': 8,
               'n_population': 8, # psutil.cpu_count(),
               'n_hof': 2,
               'verbose': True,
               'num_cpus': psutil.cpu_count()}

    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    # evolver = ProbabilityLookupEvolver(verbose=False, debug=False)
    # evolver = SizeAwareLookupEvolver(verbose=False)
    evolver = HessianProbabilityEvolver(verbose=True, debug=False)
    # evolver = OperatorBasedProbEvolver(verbose=False, debug=False, op_to_prob={AddSeriesComponent:0.5, RemoveComponent:1, SwapComponent:1, AddParallelComponent:0.5})
    nodes = {'source':TerminalSource(),
             0:VariablePowerSplitter(),
             'sink':TerminalSink()
            }
    edges = {('source', 0):ContinuousWaveLaser(),
            (0,'sink'):PhaseModulator(),
            }
    start_graph = Graph.init_graph(nodes=nodes, edges=edges)
    start_graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    # update_rules = ['random', 'preferential', 'preferential simple subpop scheme', 'preferential vectorDIFF', 'preferential photoNEAT']
    update_rules = ['tournament', 'roulette', 'preferential']

    # crossover_option = [None, crossover_maker]
    crossover_option = [None]
    for rule in update_rules:
        for cross_opt in crossover_option:
            if rule == 'random' and cross_opt is not None: # no crossover implemented in random update so far
                continue
            print(f'Starting optimization with rule {rule}, crossover_maker: {cross_opt}')
            io = handle_io()
            hof, log = topology_optimization(copy.deepcopy(start_graph), propagator, evaluator, evolver, io, ga_opts=ga_opts, local_mode=False, update_rule=rule, crossover_maker=cross_opt)
            fig, ax = plt.subplots(1, 1, figsize=[5,3])

            ax.fill_between(log['generation'].to_numpy(dtype='float'), log['best'].to_numpy(dtype='float'), log['mean'].to_numpy(dtype='float'), color='grey', alpha=0.2)
            ax.plot(log['generation'], log['best'], label='Best')
            ax.plot(log['generation'], log['mean'], label='Population mean')
            ax.plot(log['generation'], log['minimum'], color='darkgrey', label='Population minimum')
            ax.plot(log['generation'], log['maximum'], color='black', label='Population maximum')
            ax.set(xlabel='Generation', ylabel='Cost')
            ax.legend()

            io.save_fig(fig, f'topology_log_{rule.replace(" ", "_")}_with_crossover_is_{cross_opt is not None}.png')
