
import sys
sys.path.append('..')

import os
import time
import matplotlib.pyplot as plt
import pickle

import config.config as configuration

from lib.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper

if __name__ == "__main__":
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)

    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             1:PhaseModulator(parameters_from_name={'depth':9.87654321, 'frequency':12e9}),
             2:WaveShaper(),
             3:MeasurementDevice()}
    edges = [(0, 1, DispersiveFiber(parameters=[0])),
             (1, 2, DispersiveFiber(parameters=[0])),
             (2,3)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()

    #%%
    evaluator = RadioFrequencyWaveformGeneration(propagator)

    #%%
    n_runs = 200
    logs, params, scores = [], [], []

    t_start = time.time()
    for run in range(n_runs):
        print('\nStarting genetic algorithm number {}'.format(run))
        parameters_ga, score_ga, log_ga = parameters_genetic_algorithm(graph, propagator, evaluator)
        logs.append(log_ga)
        params.append(parameters_ga)
        scores.append(score_ga)
    t_ga = time.time() - t_start
    print('Time for {} genetic algorithm is {} s | {} s average per run'.format(n_runs, t_ga, t_ga / n_runs))

    #%%
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x_ticks = [i for i in range(len(params[0]))]
    model_attributes = graph.extract_attributes_to_list_experimental(['lower_bounds', 'upper_bounds', 'parameter_names'], get_location_indices=True)

    scores = [1-(s-min(scores))/max(scores) for s in scores]
    color = plt.get_cmap('Blues')

    for parameters, score in zip(params, scores):
        scale_info = zip(parameters, model_attributes['lower_bounds'], model_attributes['upper_bounds'])
        parameters_scaled = [(p - l)/u for (p, l, u) in scale_info]

        ax.plot(x_ticks, parameters_scaled, ls='', marker='o', color=color(score))

    # add colorbar
    ax.set(xlabel='Parameter', ylabel='Scaled Parameter (1 = upper bound, 0 = lower bound)',
           xticks=x_ticks, xticklabels=model_attributes['parameter_names'], xlim=[-1, len(x_ticks)+1])
    plt.show()
    plt.savefig(os.path.join(configuration.LOG_DIRECTORY, '2020_04_14__rf_awg_parameter_spread.pdf'))

    #%%
    batch_data = {}
    for i in ('graph', 'propagator', 'evaluator', 'scores', 'params', 'logs'):
        batch_data[i] = locals()[i]

    #%%
    with open(os.path.join(configuration.LOG_DIRECTORY,'2020_04_14__rf_awg.pickle'), 'wb') as handle:
        pickle.dump(batch_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


