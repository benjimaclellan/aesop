
import matplotlib.pyplot as plt
import seaborn as sns
import time
import autograd.numpy as np

import config.config as configuration

from problems.example.graph import Graph
# from problems.example.evolution_operators import EvolutionOperators
from problems.example.assets.propagator import Propagator

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

### TODO: for each model, we could prepare the function once at the beginning (especially for locked models)

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper

from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm

from lib.hessian import get_scaled_hessian, plot_eigenvectors, lha_analysis

if True:
    np.random.seed(0)

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
    principle_main()

