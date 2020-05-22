
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import itertools
import random
import os
import time
import autograd.numpy as np

import config.config as configuration

from problems.example.evaluator import Evaluator
from problems.example.graph import Graph
# from problems.example.evolution_operators import EvolutionOperators
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

### TODO: for each model, we could prepare the function once at the beginning (especially for locked models)

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.parameter_builtin import parameters_minimize
from algorithms.parameter_random_search import parameters_random_search
from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm

from lib.analysis.hessian import get_hessian, get_scaled_hessian

from problems.example.evaluator_subclasses.case_evaluator_bin import test_poly

if False:
    np.random.seed(0)

def principle_main():
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)

    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             1:PhaseModulator(parameters_from_name={'depth':9.87654321, 'frequency':12e9}),
             2:WaveShaper(),
             3:MeasurementDevice()}
    edges = [(0,1, CorningFiber(parameters=[0])),
             (1,2, CorningFiber(parameters=[0])),
             (2,3)]

    graph = Graph(nodes, edges, propagate_on_edges = True)
    graph.assert_number_of_edges()

    # %%
    evaluator = RadioFrequencyWaveformGeneration(propagator)

    # %%
    graph.sample_parameters(probability_dist='uniform', **{'triangle_width':0.1})
    parameters, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()

    plt.close('all')
    nx.draw(graph, labels = dict(zip(graph.nodes, graph.nodes)))
    plt.show()

    # %%
    print(configuration.EVOLUTION_OPERATORS)
    print(configuration.NODE_TYPES)
    print(configuration.NODE_TYPES_ALL)

    #%%
    if False:
        t1 = time.time()
        # parameters = parameters_minimize(graph, propagator, evaluator) # TODO, match output to other optimization routines
        # parameters, best_score, log = parameters_random_search(graph, propagator, evaluator)
        parameters, best_score, log = parameters_genetic_algorithm(graph, propagator, evaluator)
        t = time.time() - t1
        print('Total time: {} s'.format(t))

    #%%
    graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)

    graph.propagate(propagator)
    graph.inspect_state(propagator)

    evaluator.compare(graph, propagator)

    # #%%
    # fig, ax = plt.subplots(1,1)
    #
    # model_attributes = graph.extract_attributes_to_list_experimental(['lower_bounds', 'upper_bounds', 'parameter_names'], get_location_indices=True)
    #
    # for metric in ['min', 'max', 'avg']:
    #     ax.plot(log['gen'], log[metric], label=metric)
    # ax.legend()
    # ax.set(xlabel='Generation', ylabel='Evaluation Score')
    # plt.show()

    #%%
    exclude_locked = True
    info = graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True, exclude_locked=exclude_locked)
    hessian = get_hessian(graph, propagator, evaluator, exclude_locked=exclude_locked)

    H0 = hessian(np.array(info['parameters']))

    fig, ax = plt.subplots()
    sns.heatmap(H0)
    ax.set(xticks = list(range(len(info['parameters']))), yticks = list(range(len(info['parameters']))))
    ax.set_xticklabels(info['parameter_names'], rotation=45, ha='center')
    ax.set_yticklabels(info['parameter_names'], rotation=45, ha='right')
    plt.show()

def test_case_evaluator():
    test_poly()

if __name__ == "__main__":
    test_case_evaluator()
