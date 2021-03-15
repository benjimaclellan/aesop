
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from lib.graph import Graph
from problems.example.assets.propagator import Propagator
from lib.functions import InputOutput

from lib.minimal_save import extract_minimal_graph_info, build_from_minimal_graph_info

from algorithms.parameter_optimization import parameters_optimize

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.evaluator_subclasses.evaluator_snr import SignalNoiseRatio

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser, PulsedLaser
from problems.example.node_types_subclasses.outputs import Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, OpticalAmplifier, IntensityModulator, VariableOpticalAttenuator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink

from problems.example.assets.additive_noise import AdditiveNoise

plt.close('all')
if __name__ == "__main__":

    propagator = Propagator(window_t=2e-9, n_samples=2**14, central_wl=1.55e-6)
    evaluator = SignalNoiseRatio(propagator)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             3: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): ContinuousWaveLaser(),
             (0, 1): PhaseModulator(),
             (1, 2): VariableOpticalAttenuator(),
             (2, 3): OpticalAmplifier(),
             (3, 'sink'): Photodiode(),
             }

    graph = Graph.init_graph(nodes, edges)
    graph.update_graph()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    # %%
    AdditiveNoise.simulate_with_noise = False
    graph.propagate(propagator)
    state_signal = np.abs(graph.measure_propagator('sink'))

    AdditiveNoise.simulate_with_noise = True
    graph.propagate(propagator)
    state_noise = np.abs(graph.measure_propagator('sink'))

    noise = state_signal - state_noise
    snr = np.mean(state_noise / state_noise)

    print(f"SNR: {snr}")
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, state_noise, label='state_noise')
    ax[0].plot(propagator.t, state_signal, label='state_signal')
    ax[0].plot(propagator.t, noise, label='noise')
    ax[0].legend()

    #%%
    graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    x0, models, parameter_index, *_ = graph.extract_parameters_to_list()
    graph, x, score, log = parameters_optimize(graph, x0=x0, method='L-BFGS', verbose=True)

    #%%
    AdditiveNoise.simulate_with_noise = False
    graph.propagate(propagator)
    state_signal = np.abs(graph.measure_propagator('sink'))

    AdditiveNoise.simulate_with_noise = True
    graph.propagate(propagator)
    state_noise = np.abs(graph.measure_propagator('sink'))

    noise = state_signal - state_noise
    snr = np.mean(state_noise / state_noise)

    print(f"SNR: {snr}")
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, state_noise, label='state_noise')
    ax[0].plot(propagator.t, state_signal, label='state_signal')
    ax[0].plot(propagator.t, noise, label='noise')
    ax[0].legend()
