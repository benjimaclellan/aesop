
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
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.parameter_builtin import parameters_minimize
from algorithms.parameter_random_search import parameters_random_search
from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm


# np.random.seed(0)

if __name__ == "__main__":
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)

    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             1:PhaseModulator(parameters_from_name={'depth':9.87654321, 'frequency':12e9}),
             2:WaveShaper(),
             3:MeasurementDevice()}
    edges = [(0,1, CorningFiber(parameters=[0])),
             (1,2, CorningFiber(parameters=[0])),
             (2,3)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()

    #%%
    evaluator = RadioFrequencyWaveformGeneration(propagator)

    #%%
    n_runs = 10
    logs_ga, logs_rs = [], []
    params_ga, params_rs = [], []
    scores_ga, scores_rs = [], []

    t_start = time.time()
    for run in range(n_runs):
        print('\nStarting random search number {}'.format(run))
        parameters_rs, score_rs, log_rs = parameters_random_search(graph, propagator, evaluator)
        logs_rs.append(log_rs)
        params_rs.append(parameters_rs)
        scores_rs.append(score_rs)
    t_rs = time.time() - t_start
    print('Time for {} random search is {} s | {} s average per run'.format(n_runs, t_rs, t_rs/n_runs))

    t_start = time.time()
    for run in range(n_runs):
        print('\nStarting genetic algorithm number {}'.format(run))
        parameters_ga, score_ga, log_ga = parameters_genetic_algorithm(graph, propagator, evaluator)
        logs_ga.append(log_ga)
        params_ga.append(parameters_ga)
        scores_ga.append(score_ga)
    t_ga = time.time() - t_start
    print('Time for {} genetic algorithm is {} s | {} s average per run'.format(n_runs, t_ga, t_ga / n_runs))

    #%% we'll add HoF as a column
    for log in logs_rs + logs_ga:
        log['hof'] = log['min']
        for loc in range(len(log)):
            log.loc[loc, 'hof'] = min(log.loc[:loc, 'min'])


    #%%
    styles = {  'min':{'ls':'-', 'alpha':1},
                'avg':{'ls':'--', 'alpha':1},
                'hof':{'ls':':', 'alpha':1} }

    fig, ax = plt.subplots(1,1)
    for log in logs_rs:
        line_min_rs, = ax.plot(log['gen'], log['min'], color=sns.color_palette('Blues_r')[0], **styles['min'])
        line_avg_rs, = ax.plot(log['gen'], log['avg'], color=sns.color_palette('Blues_r')[1], **styles['avg'])
        line_hof_rs, = ax.plot(log['gen'], log['hof'], color=sns.color_palette('Blues_r')[2], **styles['hof'])

    for log in logs_ga:
        line_min_ga, = ax.plot(log['gen'], log['min'], color=sns.color_palette('Greens_r')[0], **styles['min'])
        line_avg_ga, = ax.plot(log['gen'], log['avg'], color=sns.color_palette('Greens_r')[1], **styles['avg'])
        line_hof_ga, = ax.plot(log['gen'], log['hof'], color=sns.color_palette('Greens_r')[2], **styles['hof'])

    ax.legend([line_min_ga, line_avg_ga, line_hof_ga, line_min_rs, line_avg_rs, line_hof_rs],
              ['GA Min', 'GA Avg', 'GA HoF', 'RS Min', 'RS Avg', 'RS HoF'])

    ax.set(xlabel='Generation', ylabel='Evaluation Score')
    plt.show()
    plt.savefig(os.path.join(configuration.LOG_DIRECTORY, '2020_04_14__randomsearch_vs_geneticalgorithm_parametersonly.pdf'))