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
import networkx as nx
import numpy as np
import types
import ray
from functools import wraps

import config.config as config

from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.evolver import Evolver, CrossoverMaker, StochMatrixEvolver, SizeAwareMatrixEvolver, ReinforcementMatrixEvolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.evolution_operators.evolution_operators import SwapNode

from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.evaluator_subclasses.evaluator_pulserep import PulseRepetition
from problems.example.evaluator_subclasses.evaluator_phase_sensitivity import PhaseSensitivity

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.single_path import DelayLine, IntensityModulator, ProgrammableFilter, EDFA
from problems.example.node_types_subclasses.single_path import PhaseShifter
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof, update_hof
from algorithms.topology_optimization import similarity_full_ged, similarity_reduced_ged, graph_kernel_map_to_nodetypes


plt.close('all')
if __name__ == '__main__':
    evolver = StochMatrixEvolver(verbose=False)
    propagator = Propagator(window_t=1e-9, n_samples=2 ** 14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)

    nodes = {0:ContinuousWaveLaser(),
             -1:Photodiode()}
    edges = [(0,-1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    # graph1 = evolver.random_graph(copy.deepcopy(graph), evaluator)
    # graph2 = evolver.random_graph(copy.deepcopy(graph), evaluator)
    #
    # plt.close('all')
    #
    # fig, axs = plt.subplots(1,2)
    # graph1.draw(axs[0])
    # g1 = graph_kernel_map_to_nodetypes(graph1)
    # g1.draw(axs[1], labels={node:node for node in g1.nodes})
    #
    # fig, axs = plt.subplots(1, 2)
    # graph2, _ = evolver.evolve_graph(copy.deepcopy(graph1), evaluator)
    # graph2.draw(axs[0])
    # g2 = graph_kernel_map_to_nodetypes(graph2)
    # g2.draw(axs[1], labels={node: node for node in g2.nodes})
    # sim_reduced = similarity_reduced_ged(graph1, graph2)
    # sim_full = similarity_full_ged(graph1, graph2)
    # print(f"Similarity with reduced: {sim_reduced}, similarity with full: {sim_full}")

    #%%
    n_hof, n_pop = 6, 6
    hof, pop = [], []
    for _ in range(n_hof):
        graph = evolver.random_graph(copy.deepcopy(graph), evaluator)
        hof.append((evaluator.evaluate_graph(graph, propagator), graph))
    hof.sort(key=lambda x: x[0])

    for _ in range(n_pop):
        graph = evolver.random_graph(copy.deepcopy(graph), evaluator)
        pop.append((evaluator.evaluate_graph(graph, propagator), graph))


    hof_updated = update_hof(copy.deepcopy(hof), copy.deepcopy(pop), similarity_measure='reduced_ged', threshold_value=12.0, verbose=True)
