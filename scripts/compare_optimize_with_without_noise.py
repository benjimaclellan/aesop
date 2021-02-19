
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import autograd.numpy as np

from lib.graph import Graph
from problems.example.assets.propagator import Propagator

from lib.functions import InputOutput

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.assets.additive_noise import AdditiveNoise

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import Photodiode
from problems.example.node_types_subclasses.single_path import WaveShaper, \
    IntensityModulator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink

from algorithms.parameter_optimization import parameters_optimize

plt.close('all')
if __name__ == "__main__":
    # seed = 49
    # np.random.seed(seed)
    # random.seed(seed)
    AdditiveNoise.noise_on = False

    io = InputOutput(directory='20201212_optimize_with_noise', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=False)

    propagator = Propagator(window_t=40/12e9, n_samples=2**15, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator,
                                                 target_harmonic=12e9,
                                                 target_waveform='saw',
                                                 target_amplitude=0.1)
    mod = IntensityModulator
    wave = WaveShaper

    # wave.number_of_bins = 5
    mod.step_frequency = None
    mod.max_frequency = 50.0e9
    mod.min_frequency = 1.0e9
    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             'sink': TerminalSink()}

    cw = ContinuousWaveLaser(parameters_from_name={'peak_power': 0.04})
    im = mod(parameters=[6, 12e9, 0.0, 0.0])
    edges = {('source', 0): cw,
             (0, 1, 0): im,
             (1, 2, 0): wave(),
             (2, 'sink'): Photodiode(),
             }

    graph = Graph.init_graph(nodes, edges)
    graph.update_graph()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    # graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    x0, models, parameter_index, *_ = graph.extract_parameters_to_list()

    graph, x, score, log = parameters_optimize(graph, x0=x0, method='L-BFGS+GA', log_callback=True, verbose=True)

    # io.save_object(x0, 'initial_params.pkl')
    # io.save_object(x, 'final_params.pkl')
    # io.save_object(graph, 'graph.pkl')
    # io.save_object(log, 'log.pkl')
    # io.save_object(propagator, 'propagator.pkl')
    # io.save_object(evaluator, 'evaluator.pkl')
    #
    # graph.draw()

    #%%
    graph.distribute_parameters_from_list(x, models, parameter_index)

    graph.propagate(propagator, save_transforms=False)

    evaluator.evaluate_graph(graph, propagator)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, evaluator.target)

    state = graph.measure_propagator('sink')
    ax[0].plot(propagator.t, np.abs(state))
