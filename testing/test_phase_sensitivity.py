
import sys
sys.path.append('..')

import networkx as nx
import itertools
import os
import random
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np

import config.config as configuration

from problems.example.evaluator import Evaluator
from problems.example.evolver import Evolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.evaluator_subclasses.evaluator_pulserep import PulseRepetition
from problems.example.evaluator_subclasses.evaluator_phase_sensitivity import PhaseSensitivity
from problems.example.evolver import Evolver, CrossoverMaker, StochMatrixEvolver, SizeAwareMatrixEvolver, ReinforcementMatrixEvolver

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.single_path import DelayLine, IntensityModulator, ProgrammableFilter, OpticalAmplifier
from problems.example.node_types_subclasses.single_path import PhaseShifter
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.parameter_optimization import parameters_optimize
# from algorithms.parameter_random_search import parameters_random_search
# from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm

from problems.example.assets.additive_noise import AdditiveNoise

# np.random.seed(0)
plt.close('all')
if __name__ == "__main__":
    AdditiveNoise.simulate_with_noise = False
    propagator = Propagator(window_t = 100e-9, n_samples = 2**14, central_wl=1.55e-6)
    evolver = Evolver()

    phase, phase_node = (0.5*np.pi, -2)
    phase_shifter = PhaseShifter(parameters=[phase])
    evaluator = PhaseSensitivity(propagator, phase=phase, phase_model=phase_node)
    evolver = StochMatrixEvolver()
    cw = ContinuousWaveLaser(parameters_from_name={'peak_power':1.5})
    nodes = {0:cw,
             phase_node:phase_shifter,
             1:VariablePowerSplitter(parameters=[0.5]),
             2:VariablePowerSplitter(parameters=[0.5]),
             -1:MeasurementDevice()}
    edges = [(0,1), (1,phase_node), (phase_node, 2), (1, 2), (2,-1)]
    # edges = [(0,phase_node), (phase_node,-1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    # graph, evo_op = evolver.evolve_graph(graph, evaluator)

    graph.propagate(propagator)

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


    graph.propagate(propagator, save_transforms=False)
    state = graph.measure_propagator(-1)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, power_(cw.propagate([propagator.state], propagator)[0]), label='Input', ls='-')
    ax[0].plot(propagator.t, power_(state), label='Output', ls='--')
    ax[0].plot(propagator.t, power_(evaluator.target), label='Target', ls=':')
    ax[1].plot(propagator.f, psd_(state, propagator.dt, propagator.df))
    ax[1].plot(propagator.f, psd_(evaluator.target, propagator.dt, propagator.df))
    ax[0].legend()
    #
    # graph.propagate(propagator, save_transforms=True)
    # graph.visualize_transforms([0,1,-1], propagator)
