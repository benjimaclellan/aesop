# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# ---------------------------------------------------------------

import autograd.numpy as np
import random 

from problems.example.assets.propagator import Propagator
from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator
from problems.example.node_types_subclasses.multi_path import  VariablePowerSplitter
from lib.graph import Graph
from problems.example.evolver import EGreedyHessianEvolver

from algorithms.topology_optimization import parameters_optimize_complete

"""
This is to test hessian decision making, and to tune the hyperparameters to some default values
"""

# ------------------------ General ----------------------------
np.random.seed(0)
random.seed(0)

propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
evaluator = RadioFrequencyWaveformGeneration(propagator)
evolver = EGreedyHessianEvolver(verbose=True, debug=True, epsilon=0)

# ---------------------- Test setups --------------------------
def CWL_PM_PD():
    nodes = {0:ContinuousWaveLaser(),
             1: PhaseModulator(),
             -1: Photodiode()
    }
    edges = [(0, 1), (1, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()
    _, graph = parameters_optimize_complete((None, graph), evaluator, propagator)
    return graph

def CWL_BS_PM2_BS_PD(terminal_splitter=False):
    nodes = {0:ContinuousWaveLaser(),
             1: VariablePowerSplitter(),
             2: PhaseModulator(),
             3: PhaseModulator(),
             4: VariablePowerSplitter(),
             -1: Photodiode()
    }
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()

    if terminal_splitter:
        graph.nodes[1]['model'].parameter_locks[0] = True

    _, graph = parameters_optimize_complete((None, graph), evaluator, propagator)

    if terminal_splitter:
        graph.nodes[1]['model'].parameter_locks[0] = False # unlock such that the param is considered by hessian analysis

    return graph

# ------------------------- Testing ----------------------------

def basic_freewheeling():
    graph = CWL_PM_PD()
    evolver.evolve_graph(graph, evaluator)

def freewheeling_on_remove_path():
    graph = CWL_BS_PM2_BS_PD(terminal_splitter=False)
    evolver.evolve_graph(graph, evaluator)

def terminal_path():
    graph = CWL_BS_PM2_BS_PD(terminal_splitter=True)
    evolver.evolve_graph(graph, evaluator)

# -------------------------- Main ------------------------------
if __name__=='__main__':
    basic_freewheeling()
    freewheeling_on_remove_path()
    terminal_path()
