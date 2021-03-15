
import sys
import copy
sys.path.append('..')

import matplotlib.pyplot as plt
import autograd.numpy as np

from lib.graph import Graph
from lib.functions import InputOutput

from problems.example.assets.propagator import Propagator

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.evaluator_subclasses.evaluator_pulserep import PulseRepetition

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser, PulsedLaser
from problems.example.node_types_subclasses.outputs import Photodiode, MeasurementDevice
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, OpticalAmplifier, DispersiveFiber
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink
from problems.example.assets.additive_noise import AdditiveNoise

from algorithms.parameter_optimization import parameters_optimize

plt.close('all')
# np.random.seed(0)

if __name__ == "__main__":
    AdditiveNoise.noise_on = False
    AdditiveNoise.simulate_with_noise = False
    pulse_width, rep_t, peak_power = (3e-12, 1 / 10.0e9, 0.75)
    p, q = (1, 2)
    propagator = Propagator(window_t=10*rep_t, n_samples=2 ** 15, central_wl=1.55e-6)

    input_laser = PulsedLaser(parameters_from_name={'pulse_width': pulse_width, 'peak_power': peak_power,
                                                    't_rep': rep_t, 'pulse_shape': 'gaussian',
                                                    'central_wl': 1.55e-6, 'train': True})
    input_laser.node_lock = True
    # input_laser.protected = True

    input_pulse = np.power(input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width,
                                                       rep_t=rep_t, peak_power=peak_power), 2)
    target = np.power(input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width, rep_t=rep_t * (p / q),
                                                  peak_power=peak_power * (p / q)), 2)
    evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             'sink': TerminalSink()}

    fiber1 = DispersiveFiber()
    fiber2 = DispersiveFiber()
    edges = {('source', 0): input_laser,
             (0, 1): fiber1,
             (1, 2): fiber2,
             (2, 'sink'): MeasurementDevice(),
             }

    graph = Graph.init_graph(nodes, edges)
    graph.update_graph()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    method = 'L-BFGS+PSO'

    # graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    # x0, models, parameter_index, *_ = graph.extract_parameters_to_list()
    # graph, x, score, log = parameters_optimize(graph, x0=x0, method=method, verbose=True)

    x_Talbot = copy.deepcopy(p/q * rep_t**2 / (2 * np.pi * np.abs(DispersiveFiber()._beta2_experimental)) / 2)

    ss = 0.9
    # x = [x_Talbot * ss, x_Talbot * (1-ss)]
    x = [x_Talbot, 0]

    fiber1.parameters = [x_Talbot/2]
    fiber2.parameters = [x_Talbot/2]
    # graph.func(x)
    # graph.propagate(propagator)

    x = graph.extract_parameters_to_list()[0]

    # graph.propagate(propagator)
    graph, x, score, log = parameters_optimize(graph, x0=x, method='L-BFGS', verbose=True)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(np.abs(input_pulse), ls='-', label='input')
    # ax.plot(np.abs(target), ls='--', label='target')
    # ax.plot(evaluator.shift_function(graph.measure_propagator('sink'), propagator), label='shifted-sink')
    # # ax.plot(graph.measure_propagator('sink'), ls='-.', label='unshifted-sink')
    # ax.legend()

    #%%
    # graph, x, score, log = parameters_optimize(graph, x0=x, method="L-BFGS+PSO", verbose=True)
    # input_laser.node_lock = False
    # graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=False)
    # hess = graph.hess(x)
    # # print(hess)
    # eigvals, eigvecs = np.linalg.eigh(hess)
    # print(eigvals)

    #%%
    ans = 'y' #input()
    if ans == 'y':
        io = InputOutput(directory='automatic_differentiation_system', verbose=True)
        io.init_save_dir(sub_path='talbot_two_fibers-v2', unique_id=False)

        # graph.draw()
        input_laser.node_lock = False

        io.save_object(graph, 'graph.pkl')
        io.save_object(propagator, 'propagator.pkl')
        io.save_object(evaluator, 'evaluator.pkl')
        # for (xi, model) in zip(x, models):
        #     print(f'{xi}, {model}')

