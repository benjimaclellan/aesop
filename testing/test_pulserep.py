
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

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine, ProgrammableFilter
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.parameter_builtin import parameters_optimize
# from algorithms.parameter_random_search import parameters_random_search
# from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm


# np.random.seed(0)
plt.close('all')
if __name__ == "__main__":
    propagator = Propagator(window_t = 100e-9, n_samples = 2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    evolver = Evolver()
    nodes = {0: PulsedLaser(parameters_from_name={'pulse_shape':'gaussian', 'pulse_width':10e-12,'peak_power':1,
                                                 't_rep':10e-9, 'central_wl':1.55e-6, 'train':True}),
             # 1: ProgrammableFilter(parameters=2*50*[1]),
             1: ProgrammableFilter(parameters=list(np.random.rand(10)) + list(4*np.pi*np.random.rand(10))),
             2: CorningFiber(parameters=[10]),
            -1: MeasurementDevice()}
    edges = [(0,1), (1,2), (2,-1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    method = 'L-BFGS+GA'

    # graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    # x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    # graph, x, score, log = parameters_optimize(graph, x0=x0, method=method, verbose=True)

    fig = plt.figure()
    graph.draw()

    # graph.distribute_parameters_from_list(x, node_edge_index, parameter_index)
    graph.propagate(propagator, save_transforms=True)
    state = graph.measure_propagator(-1)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, power_(state))
    ax[1].plot(propagator.f, psd_(state, propagator.dt, propagator.df))

    graph.visualize_transforms(graph.nodes, propagator)
