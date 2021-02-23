
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from lib.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_

from problems.example.evaluator_subclasses.evaluator_phase_sensitivity import PhaseSensitivity

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, OpticalAmplifier, PhaseShifter, DispersiveFiber
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink

from problems.example.assets.additive_noise import AdditiveNoise

# np.random.seed(0)
plt.close('all')
if __name__ == "__main__":
    AdditiveNoise.simulate_with_noise = False
    propagator = Propagator(window_t=100e-9, n_samples=2**14, central_wl=1.55e-6)

    PhaseShifter.protected = True

    phase, phase_node = (0.5 * np.pi, 'phase-shift')
    phase_shifter = PhaseShifter(parameters=[phase])

    evaluator = PhaseSensitivity(propagator, phase=phase, phase_model=phase_shifter)

    propagator = Propagator(window_t=10/12e9, n_samples=2**14, central_wl=1.55e-6)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0, 0): ContinuousWaveLaser(),
             (0, 1, 0): phase_shifter,
             (0, 1, 1): DispersiveFiber(parameters=[0]),
             (1, 'sink', 1): Photodiode(),
             }

    graph = Graph.init_graph(nodes, edges)
    graph.update_graph()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    graph.propagate(propagator)

    phases = np.linspace(0, 2*np.pi, 5)
    powers = np.zeros_like(phases)
    for i, phase in enumerate(phases):
        phase_shifter.parameters = [phase]
        graph.propagate(propagator)
        powers[i] = np.mean(np.abs(graph.measure_propagator('sink')))

    fig, ax = plt.subplots(1, 1)
    ax.plot(phases, powers)

    # #%%
    # method = 'L-BFGS+GA'

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
    print(evaluator.evaluate_graph(graph, propagator))


    # graph.propagate(propagator, save_transforms=False)
    # state = graph.measure_propagator(-1)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(propagator.t, power_(cw.propagate([propagator.state], propagator)[0]), label='Input', ls='-')
    # ax[0].plot(propagator.t, power_(state), label='Output', ls='--')
    # ax[0].plot(propagator.t, power_(evaluator.target), label='Target', ls=':')
    # ax[1].plot(propagator.f, psd_(state, propagator.dt, propagator.df))
    # ax[1].plot(propagator.f, psd_(evaluator.target, propagator.dt, propagator.df))
    # ax[0].legend()
    # #
    # graph.propagate(propagator, save_transforms=True)
    # graph.visualize_transforms([0,1,-1], propagator)
