
import sys
sys.path.append('..')

import time
import matplotlib.pyplot as plt
import autograd.numpy as np

import config.config as configuration

from problems.example.evaluator import Evaluator
from problems.example.evolver import Evolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.parameter_builtin import parameters_optimize

plt.close('all')
if __name__ == "__main__":
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    evolver = Evolver()
    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             1:PhaseModulator(parameters_from_name={'depth':9.87654321, 'frequency':12e9}),
             2:WaveShaper(),
             -1:MeasurementDevice()}
    edges = [(0,1),
             (1,2),
             (2,-1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    methods = ["L-BFGS", "ADAM", "GA", "CMA", "L-BFGS+GA", "ADAM+GA"]
    n_runs = 10
    comparison_results = {}
    for method in methods:
        comparison_results[method] = {'time': [], 'score': []}

    for run in range(n_runs):
        print("\n\tRun iteration: {}".format(run))
        for method in methods:
            t1 = time.time()

            graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
            x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
            graph, x, score, log = parameters_optimize(graph, x0=x0, method=method, verbose=False)

            t2 = time.time()

            comparison_results[method]['time'].append(t2-t1)
            comparison_results[method]['score'].append(score)

    fig, ax = plt.subplots(2, 2, figsize=[15, 10])
    plt.title("Based on {} runs".format(n_runs))
    ax[0,0].bar(methods, [np.mean(comparison_results[method]['time']) for method in methods])
    ax[0,0].set_ylabel('Time - Mean (s)')

    ax[0,1].bar(methods, [np.mean(comparison_results[method]['score']) for method in methods])
    ax[0,1].set_ylabel('Score - Mean')

    ax[1,0].bar(methods, [np.std(comparison_results[method]['time']) for method in methods])
    ax[1,0].set_ylabel('Time - SD (s)')

    ax[1,1].bar(methods, [np.std(comparison_results[method]['score']) for method in methods])
    ax[1,1].set_ylabel('Score - SD')


    fig = plt.figure()
    graph.draw()

    # graph.distribute_parameters_from_list(x, node_edge_index, parameter_index)
    # graph.propagate(propagator, save_transforms=False)
    # state = graph.measure_propagator(-1)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(propagator.t, np.power(np.abs(state), 2))
