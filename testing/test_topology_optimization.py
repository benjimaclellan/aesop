# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

import os
from pathlib import Path
import random
import time
import string
from datetime import date
import matplotlib.pyplot as plt
import autograd.numpy as np

from lib.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import PhaseModulator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from problems.example.evolver import HessianProbabilityEvolver

from algorithms.topology_optimization import topology_optimization
from problems.example.node_types import TerminalSource, TerminalSink

from problems.example.evolution_operators.evolution_operators import AddSeriesComponent, RemoveComponent, SwapComponent, AddParallelComponent

random.seed(10)
np.random.seed(0)
plt.close('all')


def matrix_evolver_basic_test(evolver):
    nodes = {'source':TerminalSource(),
             0:VariablePowerSplitter(),
             'sink':TerminalSink()
            }
    edges = {('source', 0):ContinuousWaveLaser(),
            (0,'sink'):PhaseModulator(),
            }
    graph = Graph.init_graph(nodes=nodes, edges=edges)
    graph.assert_number_of_edges()

    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)

    return evolver.random_graph(graph, evaluator, propagator=propagator, view_evo=True, n_evolutions=50) # doesn't actually need an evaluator in this implementation


def test_topology_opt_evolvers():
    for i in range(500):
        print(f'i: {i}')
        random.seed(i)
        np.random.seed(i)
        for evolver in [HessianProbabilityEvolver(debug=True, op_to_prob={AddSeriesComponent:0.5, RemoveComponent:0, SwapComponent:0, AddParallelComponent:0.5})]:# [ProbabilityLookupEvolver(), OperatorBasedProbEvolver(), HessianProbabilityEvolver(), SizeAwareLookupEvolver()]:
            print(f'Using {evolver.__class__.__name__}')
            graph = matrix_evolver_basic_test(evolver)
            graph.draw()
            plt.show()

def topology_optimization_test():
    directory_main = os.path.join(Path(os.getcwd()).parent, 'results')
    description = "test_runs"
    directory_rand = r"{}_{}_{}".format(date.today().strftime("%Y%m%d"),
                                        ''.join(random.choice(string.hexdigits) for x in range(4)), description)
    directory = os.path.join(directory_main, directory_rand)
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    evolver = Evolver()
    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             -1:MeasurementDevice()}
    edges = [(0,-1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()
    t1 = time.time()
    graph, score, log = topology_optimization(graph, propagator, evaluator, evolver, multiprocess=True)
    t2 = time.time()
    print('Best score of {}, total time {}'.format(score, t2-t1))
    graph.assert_number_of_edges()

    graph.draw()
    plt.savefig(os.path.join(directory, "graph.png"))


    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
    x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    graph.distribute_parameters_from_list(x0, node_edge_index, parameter_index)
    graph.propagate(propagator, save_transforms=False)
    state = graph.measure_propagator(-1)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, np.power(np.abs(state), 2))

    plt.savefig(os.path.join(directory, "time-domain.png"))


if __name__ == "__main__":
    test_topology_opt_evolvers()
    # topology_optimization_test()