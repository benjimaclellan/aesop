
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
from config import config as configuration

from lib.graph import Graph
from lib.functions import InputOutput

from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.assets.functions import psd_, power_

from simulator.fiber.evaluator_subclasses.evaluator_phase_sensitivity import PhaseSensitivity

from simulator.fiber.node_types_subclasses.inputs import ContinuousWaveLaser, PulsedLaser
from simulator.fiber.node_types_subclasses.outputs import Photodiode, MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import PhaseModulator, WaveShaper, OpticalAmplifier, PhaseShifter, DispersiveFiber
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink

from simulator.fiber.assets.additive_noise import AdditiveNoise
from algorithms.parameter_optimization import parameters_optimize

# np.random.seed(0)
plt.close('all')
if __name__ == "__main__":
    AdditiveNoise.simulate_with_noise = False

    io = InputOutput()
    io.init_save_dir(sub_path='simple_phase_sensitivity', unique_id=False)

    propagator = Propagator(window_t=10e-6, n_samples=2**14, central_wl=1.55e-6)
    # propagator = Propagator(window_t=1e-10, n_samples=2**14, central_wl=1.55e-6)

    PhaseShifter.protected = True

    phase, phase_node = (0.125 * np.pi, 'phase-shift')
    phase_shifter = PhaseShifter(parameters=[phase])

    df = DispersiveFiber(parameters=[0.0])

    evaluator = PhaseSensitivity(propagator, phase=phase, phase_model=phase_shifter)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {
        ('source', 0, 0): ContinuousWaveLaser(parameters=[0.01]),
        # ('source', 0, 0): PulsedLaser(),
        (0, 1, 0): phase_shifter,
        (0, 1, 1): DispersiveFiber(parameters=[0]),
        (0, 1, 1): df,
        (1, 'sink', 1): MeasurementDevice(),
    }

    graph = Graph.init_graph(nodes, edges)
    graph.update_graph()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    print(f'\n\nparameters starting at {x0}')
    graph.extract_attributes_to_list_experimental(['upper_bounds', 'lower_bounds'])
    graph, x, score, log = parameters_optimize(graph, x0=x0, method="L-BFGS", verbose=True)
    graph.distribute_parameters_from_list(x, node_edge_index, parameter_index)
    print(f'\n\nparameters ending at {x}')

    graph.propagate(propagator)

    phase_c = phase_shifter.parameters[0]
    power_c = np.mean(np.abs(graph.measure_propagator('sink')))
    sensitivity_c = evaluator.evaluate_graph(graph, propagator)

    phases = np.linspace(-np.pi, np.pi, 100)
    powers = np.zeros_like(phases)
    scores = np.zeros_like(phases)
    for i, phase in enumerate(phases):
        phase_shifter.parameters = [phase]
        # df.parameters = [phase]
        graph.propagate(propagator)
        powers[i] = np.mean(np.abs(graph.measure_propagator('sink')))
        scores[i] = evaluator.evaluate_graph(graph, propagator)

    #%%
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(phases/np.pi, powers, color='teal')
    axs[0].scatter(phase_c/np.pi, power_c, color='orange')
    axs[0].set(ylabel='average optical power (mW)')
    axs[1].plot(phases/np.pi, scores, color='teal', label='sweeping over phase range')
    axs[1].scatter(phase_c/np.pi, sensitivity_c, color='orange', label='optimized parameters')
    axs[1].legend()
    axs[1].set(xlabel='phase shift (rad/pi)', ylabel='sensitivity (mW/rad)')

    attributes = graph.extract_attributes_to_list_experimental(['parameters',
                                                                'lower_bounds',
                                                                'upper_bounds',
                                                                'parameter_imprecisions',
                                                                'parameter_names',
                                                                'parameter_symbols'], get_location_indices=True)

    print(graph.grad(attributes['parameters']))
    print(evaluator.evaluate_graph(graph, propagator))
    # #%%
    # method = 'L-BFGS+GA'



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
