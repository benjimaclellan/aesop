
import matplotlib.pyplot as plt
import seaborn as sns
import time
import autograd.numpy as np
import random

import config.config as configuration

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser, PulsedLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, EDFA, CorningFiber, VariableOpticalAttenuator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from problems.example.evolver import CrossoverMaker
from problems.example.evolution_operators import SwapNode

from lib.hessian import get_scaled_hessian, plot_eigenvectors, lha_analysis

if True:
    np.random.seed(0)
    random.seed(0)


def crossover_main():
    nodes0 = {0:ContinuousWaveLaser(),
              1:PhaseModulator(),
              2:WaveShaper(),
              -1:MeasurementDevice()}
    edges0 = [(0, 1), (1, 2), (2, -1)]
    graph0 = Graph(nodes0, edges0, propagate_on_edges=False)
    graph0.assert_number_of_edges()

    nodes1 = {0: PulsedLaser(),
              1: CorningFiber(),
              2: VariablePowerSplitter(),
              3: VariableOpticalAttenuator(),
              4: EDFA(),
              5: VariablePowerSplitter(),
              -1: Photodiode()
             }
    edges1 = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, -1)]
    graph1 = Graph(nodes1, edges1, propagate_on_edges=False)

    graph0.draw()
    graph1.draw()
    plt.show()

    crossover_maker = CrossoverMaker()
    for _ in range(5):
        graph0, graph1, _ = crossover_maker.crossover_graphs(graph0, graph1)
        # graph0.draw()
        # graph1.draw()
        # plt.show()

def max_output_main():
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    nodes = {0:ContinuousWaveLaser(),
             1:PhaseModulator(),
             2:WaveShaper(),
             3:EDFA(parameters_from_name={'max_small_signal_gain_dB':50}),
            -1:Photodiode()
            }
    edges = [(0, 1), (1, 2), (2, 3), (3, -1)]

    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()
    graph.propagate(propagator)
    graph.inspect_state(propagator)


def principle_main():
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             1:PhaseModulator(parameters_from_name={'depth':9.87654321, 'frequency':12e9}),
             2:WaveShaper(),
             3:MeasurementDevice()}
    edges = [(0,1), (1,2), (2,3)]


    graph = Graph(nodes, edges, propagate_on_edges=False)
    graph.assert_number_of_edges()

    # %%
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    print("evaluator target rf shape: {}".format(evaluator.target_f.shape))
    print("evaluator scale array shape: {}".format(evaluator.target_f.shape))
    evaluator.evaluate_graph(graph, propagator)

    # %%
    graph.sample_parameters(probability_dist='uniform', **{'triangle_width':0.1})
    parameters, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()

    plt.close('all')

    # nx.draw(graph, labels = dict(zip(graph.nodes, graph.nodes)))
    graph.draw(labels = dict(zip(graph.nodes, graph.nodes)))
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

    graph.propagate(propagator, save_transforms=True)
    # graph.inspect_state(propagator)
    state = graph.measure_propagator(2)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, np.power(np.abs(state),2))

    evaluator.compare(graph, propagator)

    graph.visualize_transforms(nodes_to_visualize=graph.nodes, propagator=propagator)


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
    hessian = get_scaled_hessian(graph, propagator, evaluator, exclude_locked=exclude_locked)

    (H_diag, H, H_evals, H_evecs) = lha_analysis(hessian, info['parameters'])

    fig, ax = plt.subplots()
    sns.heatmap(H)
    ax.set(xticks = [i+0.5 for i in range(len(info['parameters']))], yticks = [i+0.5 for i in range(len(info['parameters']))])
    ax.set_xticklabels(info['parameter_names'], rotation=45, ha='center', va='top')
    ax.set_yticklabels(info['parameter_names'], rotation=45, ha='right', va='center')
    plt.show()


    fig, ax = plt.subplots()
    test_mat = np.zeros([len(H_evals), len(H_evals) + 1])
    for kk, (eval, evec) in enumerate(sorted(zip(H_evals, H_evecs), key=lambda x:x[0])):
        test_mat[kk, 1:] = evec/np.max(np.abs(evec))
        test_mat[kk, 0] = eval/np.max(np.abs(H_evals))
    sns.heatmap(test_mat)
    ax.set(xticks=[i + 0.5 for i in range(len(info['parameters'])+1)],
           yticks=[i + 0.5 for i in range(len(info['parameters']))])
    ax.set_xticklabels(['Eigenvalue'] + info['parameter_names'], rotation=45, ha='center', va='top')
    ax.set_yticklabels(['{:1.1e}'.format(H_evals[i]) for i in range(len(H_evals))], rotation=45, ha='right', va='center')
    plt.show()

    ax = plot_eigenvectors(info['parameter_names'], H_evecs, H_evals)


if __name__ == "__main__":
    # principle_main()
    # crossover_main()
    max_output_main()
    # evolution_tests_main()
