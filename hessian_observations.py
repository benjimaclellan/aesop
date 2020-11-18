import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
import random

from problems.example.assets.propagator import Propagator
from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser, PulsedLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, OpticalAmplifier, DispersiveFiber, VariableOpticalAttenuator, IntensityModulator, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.graph import Graph

from problems.example.assets.hessian_graph_analysis import get_all_node_scores

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


def partial_hessian_node(h, graph, nodes):
    """
    Returns partial hessian, param names (in partial hessian)
    """
    if type(nodes) != list: # like that, nodes can be a single node, or a list of nodes
        nodes = [nodes]

    _, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    hess_node_inds = [i for (i, node_i) in enumerate(node_edge_index) if node_i in nodes]
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


def free_wheeling_thresholding(graph, free_wheeling_node, test_title='', free_wheeler_title=''):
    _, graph = parameters_optimize_complete((None, graph), evaluator, propagator)
    # confirm_free_wheeling_WS(graph)
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
    
    x, _, _, *_ = graph.extract_parameters_to_list()
    params = graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True, exclude_locked=True)
    
    # gradient should be close to zero. Unless you're at a boundary, in which case that's fine I guess
    print(f"CW params: {graph.nodes[0]['model'].parameters}")
    print(f'gradient: {graph.grad(x)}')

    free_nodes, node_edge_index, parameter_index = get_all_free_wheeling_nodes_params(graph)
    print(f'free wheeling\nnodes: {free_nodes}\nnode_edge_index: {node_edge_index}\nparameter_index: {parameter_index}')

    hessian = graph.scaled_hess(np.array(x))
    phessian, phess_params = partial_hessian_node(hessian, graph, free_wheeling_node)
    plot_hessian(phessian, phess_params, title=f'{test_title}, {free_wheeler_title}')
    plot_hessian(hessian, params, title=f'{test_title}, all')

    # # to find appropriate epsilon threshold...
    # sorted = np.sort(np.abs(hessian).flatten())[::-1]
    # print(f'sorted: {sorted[0:150]}')

    epsilon = 1e-4
    bin_hess = binarize_hessian(hessian, epsilon)
    plot_hessian(bin_hess, params, title=f'{test_title}, binarized')
    bin_phess = binarize_hessian(phessian, epsilon)
    plot_hessian(bin_phess, phess_params, title=f'{test_title}, binarized, {free_wheeler_title}')


def linked_parameter_checks(graph, title=''):
    _, graph = parameters_optimize_complete((None, graph), evaluator, propagator)
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
    
    params = graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True, exclude_locked=True)
    H_diag, H, H_evals, H_evecs = lha_analysis(graph.scaled_hess, params['parameters'])

    print(f'H: {H}')
    print(f'H_diag: {H_diag}')
    plot_hessian(H, params, title=f'{title}, original')
    print(f'Eigenvalues: {H_evals}')
    print(f'Eigenvectors: {H_evecs}')

    print(f"param names: {params['parameter_names']}")
    print(f'eigenvalues (smallest 3): {H_evals[0:3]}')
    print(f'eigenvectors (matching): {H_evecs[:, 0:3]}')


def print_node_freewheeling_terminal_scores(graph):
    _, graph = parameters_optimize_complete((None, graph), evaluator, propagator)
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    terminal_node_scores, free_wheeling_node_scores = get_all_node_scores(graph)
    print(f'terminal node score:\n{terminal_node_scores}')
    print(f'free_wheeling_node_scores:\n{free_wheeling_node_scores}\n\n')


def free_wheeling_params_0():
    nodes = {0:ContinuousWaveLaser(),
             1: PhaseModulator(),
             2: WaveShaper(),
             -1: MeasurementDevice()
    }
    edges = [(0, 1), (1, 2), (2, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()

    # free_wheeling_thresholding(graph, 2, test_title='free-wheeling params 0', free_wheeler_title='WS only')

    print_node_freewheeling_terminal_scores(graph)

def free_wheeling_params_1():
    nodes = {0:ContinuousWaveLaser(),
             1: PhaseModulator(),
             -1: MeasurementDevice()
    }
    edges = [(0, 1), (1, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()

    # free_wheeling_thresholding(graph, 1, test_title='free-wheeling params/node 1', free_wheeler_title='PM only')
    print_node_freewheeling_terminal_scores(graph)


def free_wheeling_node_2():
    splitter = VariablePowerSplitter(parameters_from_name={'coupling_ratio': 0})
    splitter.parameter_locks[0] = True # lock this such that one of the modulators is guaranteed to be blocked out

    nodes = {0:ContinuousWaveLaser(),
             1: splitter,
             2: IntensityModulator(),
             3: IntensityModulator(),
             4: VariablePowerSplitter(),
             -1: MeasurementDevice()
    }
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()

    # free_wheeling_thresholding(graph, 3, test_title='free-wheeling node 2', free_wheeler_title='disconnected IM')
    print_node_freewheeling_terminal_scores(graph)


def linked_parameters_delays():
    nodes = {0:ContinuousWaveLaser(),
             1: DelayLine(),
             2: DelayLine(),
             -1: MeasurementDevice()
    }
    edges = [(0, 1), (1, 2), (2, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()
    linked_parameter_checks(graph)


def linked_parameter_waveshapers():
    nodes = {0:ContinuousWaveLaser(),
             1: PhaseModulator(),
             2: WaveShaper(),
             3: WaveShaper(),
             -1: MeasurementDevice()
    }
    edges = [(0, 1), (1, 2), (2, 3), (3, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()
    linked_parameter_checks(graph)


if __name__=='__main__':
    # linked_parameter_waveshapers()
    # linked_parameters_delays()
    free_wheeling_params_0()
    free_wheeling_params_1()
    free_wheeling_node_2()



# def setup_for_reduction():
#     nodes = {0:ContinuousWaveLaser(),
#              1: PhaseModulator(),
#              2: CorningFiber(),
#              3: CorningFiber(),
#              4: WaveShaper(),
#              -1: Photodiode()
#     }
#     edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, -1)]
#     graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
#     graph.assert_number_of_edges()

#     run_setup(graph)


# def run_setup(graph):
#     _, graph = parameters_optimize_complete((None, graph), evaluator, propagator)
#     graph.display_noise_contributions(propagator)
#     params = graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True, exclude_locked=True)
#     H_diag, H, H_evals, H_evecs = lha_analysis(graph.hess, params['parameters'])
#     plot_hessian(H, params, title="Hessian")