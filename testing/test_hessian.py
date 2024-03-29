
import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np

from lib.graph import Graph
# from problems.example.evolution_operators import EvolutionOperators
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.outputs import MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import DispersiveFiber, WaveShaper
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter

# from algorithms.parameter_random_search import parameters_random_search
# from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink

from lib.hessian import get_hessian, lha_analysis

if True:
    np.random.seed(0)

if __name__ == "__main__":
    plt.close('all')

    propagator = Propagator(window_t=1e-7, n_samples=2 ** 14, central_wl=1.55e-6)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             3: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): PulsedLaser(parameters_from_name={'pulse_shape': 'gaussian',
                                                              'pulse_width': 1.0e-10,
                                                              'peak_power': 1,
                                                              't_rep': 1e-9, 'central_wl': 1.55e6, 'train': True}),
             (0,1): DispersiveFiber(parameters=[50]),
             (1,2): DispersiveFiber(parameters=[10]),
             (2,3): WaveShaper(),
             (3,'sink'): MeasurementDevice()}

    graph = Graph(nodes, edges)
    graph.assert_number_of_edges()

    # %%
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    # evaluator.evaluate_graph(graph, propagator)

    # %%
    graph.sample_parameters(probability_dist='uniform', **{'triangle_width':0.1})
    parameters, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)



    graph.draw(labels = dict(zip(graph.nodes, graph.nodes)))
    plt.show()

    #%%

    graph.propagate(propagator)
    # graph.inspect_state(propagator)
    state = graph.measure_propagator('sink')
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, np.power(np.abs(state),2))

    # graph.visualize_transforms(nodes_to_visualize=graph.nodes, propagator=propagator)

    #%%
    exclude_locked = True
    info = graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True)
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


