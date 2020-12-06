
import sys
sys.path.append('..')

import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import winsound
from scipy.interpolate import interp1d

import config.config as configuration

from problems.example.evaluator import Evaluator
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from lib.functions import InputOutput

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper, OpticalAmplifier, IntensityModulator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types import TerminalSource, TerminalSink

from algorithms.parameter_optimization import parameters_optimize

plt.close('all')
if __name__ == "__main__":
    random_seed = 1
    np.random.seed(random_seed)

    propagator = Propagator(window_t=10/12e9, n_samples=2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9, target_waveform='saw')
    io = InputOutput(directory='20201205_param_opt_comparison', verbose=True)
    io.init_save_dir(sub_path=f'seed_{random_seed}__', unique_id=True)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): ContinuousWaveLaser(),
             (0, 1): PhaseModulator(),
             (1, 2): WaveShaper(),
             (2, 'sink'): Photodiode(),
             }

    graph = Graph.init_graph(nodes, edges)
    graph.update_graph()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    methods = ["L-BFGS", "ADAM", "GA", "L-BFGS+GA", "ADAM+GA", "PSO", "L-BFGS+PSO"]
    # methods = ["L-BFGS"]
    palette = sns.color_palette('colorblind')
    colors = {method: color for method, color in zip(methods, palette)}

    n_runs = 5
    starting_points = []
    for run in range(n_runs):
        graph.sample_parameters(probability_dist='uniform')
        x0, models, parameter_index, *_ = graph.extract_parameters_to_list()
        starting_points.append(x0)

    optimize = True
    if optimize:
        results = {method: [] for method in methods}
        for run, starting_point in enumerate(starting_points):
            for method in methods:
                print(f"Run: {run}/{n_runs} | Method: {method}")
                graph, x, score, log = parameters_optimize(graph, x0=starting_point, method=method,
                                                           log_callback=True, verbose=False)
                results[method].append(log)
        io.save_object(results, 'results.pkl')
    else:
        results = io.load_object('results.pkl')

    #%%
    fig, ax = plt.subplots(1, 1, figsize=[8, 5])
    ax.set(xlabel='Process CPU time (s)', ylabel='Evaluation Score (a.u.)')
    for method, logs in results.items():
        for k, log in enumerate(logs):
            ax.plot(log.dataframe['process runtime (s)'], log.dataframe['mean'], label=method if k == 0 else None,
                    color=colors[method], alpha=0.8)
    ax.legend()
    io.save_fig(fig, 'results_all')

    #%%
    fig, ax = plt.subplots(1, 1, figsize=[8, 5])
    ax.set(xlabel='Process CPU time (s)', ylabel='Evaluation Score (a.u.)')
    for method, logs in results.items():
        t_max, t_min = np.inf, 0
        for k, log in enumerate(logs):
            if np.max(log.dataframe['process runtime (s)']) < t_max:
                t_max = np.max(log.dataframe['process runtime (s)'])
        t = np.linspace(t_min, t_max, 1000)

        ys = np.zeros([t.shape[0], n_runs])
        for k in range(n_runs):
            log = logs[k]
            f = interp1d(log.dataframe['process runtime (s)'].to_numpy(), log.dataframe['mean'].to_numpy(),
                         bounds_error=False, fill_value=np.nan)
            ys[:, k] = f(t)
        yavg = np.nanmean(ys, axis=1)
        ymin = np.nanmin(ys, axis=1)
        ymax = np.nanmax(ys, axis=1)
        yvar = np.sqrt(np.nanvar(ys, axis=1))

        ax.plot(t, yavg, color=colors[method], alpha=1.0, label=method)
        ax.fill_between(t, ymax, ymin, color=colors[method], alpha=0.5)
        # ax.fill_between(t, yavg-yvar, yavg+yvar, color=colors[method], alpha=0.5)

    ax.legend()
    io.save_fig(fig, 'results_condensed')

    #%%
    # winsound.Beep(440, 3000)  # alert that it is finished
