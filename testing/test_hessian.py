
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import itertools
import random
import os
import time
import autograd.numpy as np

from problems.example.evaluator import Evaluator
from problems.example.graph import Graph
# from problems.example.evolution_operators import EvolutionOperators
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.evaluator_subclasses.evaluator_power import PeakPower

### TODO: for each model, we could prepare the function once at the beginning (especially for locked models)

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.parameter_random_search import parameters_random_search
from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm

from lib.analysis.hessian import get_hessian, get_scaled_hessian, plot_eigenvectors, lha_analysis

if True:
    np.random.seed(0)

if __name__ == "__main__":
    plt.close('all')

    propagator = Propagator(window_t=1e-7, n_samples=2 ** 14, central_wl=1.55e-6)
    nodes = {0: PulsedLaser(parameters_from_name={'pulse_shape':'gaussian',
                                                  'pulse_width':1.0e-10,
                                                  'peak_power':1,
                                                  't_rep':1e-9, 'central_wl':1.55e6, 'train':True}),
             1: CorningFiber(parameters=[50]),
             2: CorningFiber(parameters=[10]),
             3: WaveShaper(),
             4: MeasurementDevice()}
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

    # propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    # nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
    #          1:PhaseModulator(parameters_from_name={'depth':9.87654321, 'frequency':12e9}),
    #          2:WaveShaper(),
    #          3:MeasurementDevice()}
    # edges = [(0,1), (1,2), (2,3)]

    # propagator = Propagator(window_t = 1e-9, n_samples = 2**16, central_wl=1.55e-6)
    # nodes = {0:PulsedLaser(parameters_from_name={'pulse_shape':'gaussian',
    #                                              'pulse_width':0.1e-12,
    #                                              'peak_power':1,
    #                                              't_rep':1e-9, 'central_wl':1.55e6, 'train':False}),
    #          1:DelayLine(parameters=[1,0.5,0.5,1]),
    #          2:MeasurementDevice()}
    # edges = [(0,1), (1,2)]

    graph = Graph(nodes, edges, propagate_on_edges=False, deep_copy=False)
    graph.assert_number_of_edges()

    # %%
    evaluator = PeakPower(propagator)
    # evaluator.evaluate_graph(graph, propagator)

    # %%
    graph.sample_parameters(probability_dist='uniform', **{'triangle_width':0.1})
    parameters, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)



    graph.draw(labels = dict(zip(graph.nodes, graph.nodes)))
    plt.show()

    #%%

    graph.propagate(propagator, save_transforms=True)
    # graph.inspect_state(propagator)
    state = graph.measure_propagator(2)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, np.power(np.abs(state),2))

    # graph.visualize_transforms(nodes_to_visualize=graph.nodes, propagator=propagator)

    #%%
    exclude_locked = True
    info = graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True, exclude_locked=exclude_locked)
    hessian = get_hessian(graph, propagator, evaluator, exclude_locked=exclude_locked)
    # hessian = get_scaled_hessian(graph, propagator, evaluator, exclude_locked=exclude_locked)

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


    # ax = plot_eigenvectors(info['parameter_names'], H_evecs, H_evals)


