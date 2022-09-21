
import sys
sys.path.append('..')

import matplotlib.pyplot as plt

from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evaluator_subclasses.evaluator_pulserep import PulseRepetition

from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.outputs import MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper, OpticalAmplifier
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink

from algorithms.parameter_optimization import parameters_optimize

plt.close('all')
if __name__ == "__main__":
    propagator = Propagator(window_t=4e-9, n_samples=2 ** 15, central_wl=1.55e-6)

    pulse_width, rep_t, peak_power = (3e-12, 1/12.0e9, 1.0)
    p, q = (1, 2)

    input_laser = PulsedLaser(parameters_from_name={'pulse_width': pulse_width, 'peak_power': peak_power,
                                                    't_rep': rep_t, 'pulse_shape': 'gaussian',
                                                    'central_wl': 1.55e-6, 'train': True})
    input_laser.node_lock = True
    input_laser.protected = True

    input = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)
    target = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width, rep_t=rep_t * (p / q),
                                         peak_power=peak_power * (p / q))
    evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)

    md = MeasurementDevice()
    md.protected = True

    pm = PhaseModulator(parameters_from_name={'depth':10, 'frequency':1e9, 'shift':0.0})
    df = DispersiveFiber(parameters_from_name={'length': 10e3})
    ws = WaveShaper()
    # beta2 = df._beta2_experimental
    # zT = (rep_t ** 2) / np.abs(2 * np.pi * beta2) * (p/q)
    # print(f'zt {zT/1e3}')
    # df.parameters = [zT]

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             3: VariablePowerSplitter(),
             'sink': TerminalSink()}
    edges = {('source', 0): input_laser,
             (0, 1): pm,
             (1, 2): df,
             (2, 3): OpticalAmplifier(),
             # (0, 1): ws,
             (3, 'sink'): md,
             }

    graph = Graph.init_graph(nodes, edges)
    graph.update_graph()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    method = 'L-BFGS+GA'

    graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    x, models, parameter_index, *_ = graph.extract_parameters_to_list()
    graph, x, score, log = parameters_optimize(graph, x0=x, method=method, verbose=True)

    # graph.draw()

    graph.distribute_parameters_from_list(x, models, parameter_index)
    graph.propagate(propagator, save_transforms=False)
    state = graph.measure_propagator('sink')
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, evaluator.target, label='Target')
    ax[0].plot(propagator.t, state, ls='-.', label='Generated')
    ax[0].plot(propagator.t, input, ls=':', label='Input')
    ax[0].legend()
    # print('Score {}\nParameters {}'.format(score, x))
    # evaluator.compare(graph, propagator)

    for (xi, model) in zip(x, models):
        print(f'{xi}, {model}')
