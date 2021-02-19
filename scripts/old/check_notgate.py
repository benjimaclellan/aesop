import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import autograd.numpy as np

from lib.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.evaluator_subclasses.evaluator_notgate import OpticalNotGate

from problems.example.assets.additive_noise import AdditiveNoise

from problems.example.node_types_subclasses.inputs import BitStream
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import WaveShaper

from algorithms.parameter_optimization import parameters_optimize

#%%
plt.close('all')
if __name__ == "__main__":
    propagator = Propagator(window_t = 1000e-9, n_samples = 2**14, central_wl=1.55e-6)
    bit_stream = BitStream(parameters=[99, 1.0, 10e-9, 1000e-12, 1.55e-6, 55.0, 0.1e3])
    print(bit_stream.bits)
    pattern = bit_stream.make_binary_pulse_pattern(propagator, bit_stream.bits)
    # pattern = bit_stream.make_binary_pulse_pattern(propagator, len(bit_stream.bits) * [1])
    bit_stream.pattern = pattern
    AdditiveNoise.simulate_with_noise = False

    target = bit_stream.make_binary_pulse_pattern(propagator, np.logical_not(bit_stream.bits).astype('int'))
    # target = pattern
    evaluator = OpticalNotGate(propagator, target=target)

    WaveShaper.frequency_bin_width = 1e8
    WaveShaper.number_of_bins = 30
    ws = WaveShaper()

    nodes = {0:bit_stream,
             1:ws,
             -1:MeasurementDevice()}
    edges = [(0,1),
             (1,-1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    # x, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    graph, x, score, log = parameters_optimize(graph, x0=x0, method='L-BFGS+GA', verbose=True)
    # fig = plt.figure()
    graph.draw()

    graph.distribute_parameters_from_list(x, node_edge_index, parameter_index)
    graph.propagate(propagator, save_transforms=True)
    graph.visualize_transforms([1], propagator)

    state = graph.measure_propagator(-1)

    score = evaluator.evaluate_graph(graph, propagator)

    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(propagator.t, np.power(np.abs(state), 2))
    print('Score {}\nParameters {}'.format(score, x))
    evaluator.compare(graph, propagator)


    # xs, scores = [], []
    # for i in range(500):
    #     delay = 1e-9 * i
    #     graph.distribute_parameters_from_list([delay], node_edge_index, parameter_index)
    #     graph.propagate(propagator, save_transforms=False)
    #     state = graph.measure_propagator(-1)
    #
    #     score = evaluator.evaluate_graph(graph, propagator)
    #
    #     scores.append(score)
    #     xs.append(delay)
    #
    # fig, ax = plt.subplots(1,1)
    # ax.plot(xs, scores)