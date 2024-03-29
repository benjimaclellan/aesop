
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import autograd.numpy as np

from simulator.fiber.evolver import Evolver
from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evaluator_subclasses.evaluator_pulserep import PulseRepetition

from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.outputs import MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import PhaseModulator, IntensityModulator, OpticalAmplifier

# from algorithms.parameter_random_search import parameters_random_search
# from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm


# np.random.seed(0)
plt.close('all')
if __name__ == "__main__":
    propagator = Propagator(window_t = 100e-9, n_samples = 2**14, central_wl=1.55e-6)
    evolver = Evolver()

    pulse_width, rep_t, peak_power = (0.2e-9, 20.0e-9, 1.0)
    p, q = (2,1)

    pl = PulsedLaser(parameters_from_name={'pulse_width':pulse_width,'peak_power':peak_power, 't_rep':rep_t,
                                           'pulse_shape':'gaussian', 'central_wl':1.55e-6, 'train':True})
    pl.node_lock = True
    input = pl.get_pulse_train(propagator.t, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)
    target = pl.get_pulse_train(propagator.t, pulse_width=pulse_width*(p/q), rep_t=rep_t*(p/q), peak_power=peak_power*(p/q))
    evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)

    nodes = {0:pl,
             1:IntensityModulator(parameters_from_name={'depth':np.pi*0.5, 'frequency':1/rep_t/2, 'bias':np.pi*0.5}),
             2:OpticalAmplifier(),
             -2:PhaseModulator(),
            -1: MeasurementDevice()}
    edges = [(0,1), (1,2), (2,-2), (-2,-1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    # #%%
    method = 'L-BFGS+GA'

    # graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    # x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    # print(f'\n\nparameters starting at {x0}')
    # graph.extract_attributes_to_list_experimental(['upper_bounds', 'lower_bounds'])
    # graph, x, score, log = parameters_optimize(graph, x0=x0, method=method, verbose=True)
    # graph.distribute_parameters_from_list(x, node_edge_index, parameter_index)
    # print(f'\n\nparameters ending at {x}')

    # fig = plt.figure()
    # graph.draw()
    #
    #
    # # x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    # # delays = np.linspace(0, 20e-9, 50)
    # # scores = []
    # # for delay in delays:
    # #     graph.nodes[2]['model'].set_parameter_from_name('delay', delay)
    # #     graph.propagate(propagator, save_transforms=True)
    # #     scores.append(evaluator.evaluate_graph(graph, propagator))
    # # fig, ax = plt.subplots(1,1)
    # # ax.plot(delays, scores)
    #
    graph.propagate(propagator, save_transforms=True)
    # state = graph.measure_propagator(-1)
    # print(evaluator.evaluate_graph(graph, propagator))
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(propagator.t, power_(input), label='Input', ls='-')
    # ax[0].plot(propagator.t, power_(state), label='Output', ls='--')
    # ax[0].plot(propagator.t, power_(evaluator.target), label='Target', ls=':')
    # ax[1].plot(propagator.f, psd_(state, propagator.dt, propagator.df))
    # ax[1].plot(propagator.f, psd_(evaluator.target, propagator.dt, propagator.df))
    # ax[0].legend()
    #
    # graph.propagate(propagator, save_transforms=True)
    # graph.visualize_transforms([0,1,-1], propagator)
