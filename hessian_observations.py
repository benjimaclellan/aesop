import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
import random

from problems.example.assets.propagator import Propagator
from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser, PulsedLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, EDFA, CorningFiber, VariableOpticalAttenuator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.graph import Graph

from lib.hessian import get_scaled_hessian, plot_eigenvectors, lha_analysis
from algorithms.topology_optimization import parameters_optimize_complete

from problems.example.assets.functions import psd_

"""
Just to learn about things and look at effects

1. Try 2 setups that clearly could be reduced into one (i.e. 2 delay lines)
   Tests grouping components (lock together)
2. 2 setups that can less obviously be reduced (e.g. two waveshapers in a row)
   Tests same as 1
3. One setup where a certain component is doing nothing (e.g. beam splitter that's entirely biased to one side, the other branch is blank)
   Tests mechanism for potential path deletion
"""

np.random.seed(0)
random.seed(0)

propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
evaluator = RadioFrequencyWaveformGeneration(propagator)
# np.set_printoptions(precision=1) # make matrices a little more manageable to print

def plot_hessian(H, info, title=''):
    fig, ax = plt.subplots()
    sns.heatmap(H)
    ax.set(xticks = [i+0.5 for i in range(len(info['parameter_names']))], yticks = [i+0.5 for i in range(len(info['parameter_names']))])
    ax.set_xticklabels(info['parameter_names'], rotation=45, ha='center', va='top')
    ax.set_yticklabels(info['parameter_names'], rotation=45, ha='right', va='center')
    plt.title(title)
    plt.show()

def confirm_free_wheeling_WS(graph):
    graph.propagate(propagator, save_transforms=True)
    PM_output_psd = psd_(graph.measure_propagator(1), propagator.dt, propagator.df)
    bins = graph.nodes[2]['model'].transform[1]

    _, ax = plt.subplots()
    ax.plot(propagator.f, PM_output_psd / np.max(PM_output_psd), label='PM output')
    ax.plot(propagator.f, bins[1], label='frequency bins amplitude')
    # ax.plot(propagator.f, bins[4], label='frequency bins amp')
    ax.legend()
    plt.show()

def lock_node_params(graph, nodes):
    for node in nodes:
        for i in range(graph.nodes[node]['model'].number_of_parameters):
            graph.nodes[node]['model'].parameter_locks[i] = True


def partial_hessian(h, inds):
    ph = h[np.ix_(inds, inds)]
    return ph


def partial_hessian_node(h, graph, node):
    """
    Returns partial hessian, param names (in partial hessian)
    """
    _, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    hess_node_inds = [i for (i, node_i) in enumerate(node_edge_index) if node_i == node]
    params = graph.extract_attributes_to_list_experimental(['parameter_names'], get_location_indices=True, exclude_locked=True)['parameter_names']
    params = {'parameter_names': [params[i] for i in hess_node_inds]}

    return partial_hessian(h, hess_node_inds), params

def binarize_hessian(H, thresh):
    bin_H = np.ones_like(H)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.abs(H[i, j]) > thresh:
                bin_H[i, j] = 1
            else:
                bin_H[i, j] = 0
    return bin_H


def free_wheeling_params_0():
    nodes = {0:ContinuousWaveLaser(),
             1: PhaseModulator(),
             2: WaveShaper(),
             -1: MeasurementDevice()
    }
    edges = [(0, 1), (1, 2), (2, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()

    _, graph = parameters_optimize_complete((None, graph), evaluator, propagator)
    # confirm_free_wheeling_WS(graph)
    locked_nodes = [0, 1, -1]
    # lock_node_params(graph, locked_nodes)
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
    
    x, _, _, *_ = graph.extract_parameters_to_list()
    params = graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True, exclude_locked=True)
    print(f'params: {params}')
    hessian = graph.scaled_hess(np.array(x))
    phessian, phess_params = partial_hessian_node(hessian, graph, 2)
    print(f'phess params: {phess_params}')
    plot_hessian(phessian, phess_params, title='floating params 0, WS only')
    plot_hessian(hessian, params, title='floating params 0, all')

    # # to find appropriate epsilon threshold...
    # sorted = np.sort(np.abs(hessian).flatten())[::-1]
    # print(f'sorted: {sorted[0:150]}')

    epsilon = 1e-4
    bin_hess = binarize_hessian(hessian, epsilon)
    plot_hessian(bin_hess, params, title='floating params 0, binarized')
    bin_phess = binarize_hessian(phessian, epsilon)
    plot_hessian(bin_phess, phess_params, title='floating params 0, binarized, WS only')


if __name__=='__main__':
    free_wheeling_params_0()
    # archetypical_design()
    # setup_for_reduction()


def setup_for_reduction():
    nodes = {0:ContinuousWaveLaser(),
             1: PhaseModulator(),
             2: CorningFiber(),
             3: CorningFiber(),
             4: WaveShaper(),
             -1: Photodiode()
    }
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()

    run_setup(graph)


def run_setup(graph):
    _, graph = parameters_optimize_complete((None, graph), evaluator, propagator)
    graph.display_noise_contributions(propagator)
    params = graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True, exclude_locked=True)
    H_diag, H, H_evals, H_evecs = lha_analysis(graph.hess, params['parameters'])
    plot_hessian(H, params, title="Hessian")