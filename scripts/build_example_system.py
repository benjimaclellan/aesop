
import sys
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

from algorithms.parameter_optimization import parameters_optimize

plt.close('all')
if __name__ == "__main__":
    io = InputOutput(directory='automatic_differentiation_system', verbose=True)
    io.init_save_dir(sub_path='talbot_two_fibers', unique_id=False)

    propagator = Propagator(window_t=4e-9, n_samples=2 ** 15, central_wl=1.55e-6)
    pulse_width, rep_t, peak_power = (3e-12, 1 / 10.0e9, 1.0)
    p, q = (1, 2)

    input_laser = PulsedLaser(parameters_from_name={'pulse_width': pulse_width, 'peak_power': peak_power,
                                                    't_rep': rep_t, 'pulse_shape': 'gaussian',
                                                    'central_wl': 1.55e-6, 'train': True})
    input_laser.node_lock = True
    # input_laser.protected = True

    input = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)
    target = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width * (p / q), rep_t=rep_t * (p / q),
                                         peak_power=peak_power * (p / q))
    evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): input_laser,
             (0, 1): DispersiveFiber(),
             (1, 2): DispersiveFiber(),
             (2, 'sink'): MeasurementDevice(),
             }

    graph = Graph.init_graph(nodes, edges)
    graph.update_graph()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    method = 'L-BFGS+GA'

    graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    x0, models, parameter_index, *_ = graph.extract_parameters_to_list()
    graph, x, score, log = parameters_optimize(graph, x0=x0, method=method, verbose=True)

    graph.draw()
    input_laser.node_lock = False

    io.save_object(graph, 'graph.pkl')
    io.save_object(propagator, 'propagator.pkl')
    io.save_object(evaluator, 'evaluator.pkl')
    for (xi, model) in zip(x, models):
        print(f'{xi}, {model}')
