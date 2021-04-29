"""
Test of topology optimization routines

TODO: HoF should be taken prior to speciation!!!!!
"""

# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform
import copy

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# various imports
import matplotlib.pyplot as plt
import psutil
import autograd.numpy as np
from config import config
from lib.functions import InputOutput

from problems.example.evolver import HessianProbabilityEvolver, OperatorBasedProbEvolver
from lib.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.evaluator_subclasses.evaluator_noise_suppression import NoiseSuppression

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser, NoisySignal
from problems.example.node_types_subclasses.outputs import Photodiode, MeasurementDevice, ElectricFieldMeasurementDevice
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, OpticalAmplifier, \
    DispersiveFiber, DelayLine, PhaseShifter
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink
from problems.example.assets.additive_noise import AdditiveNoise
from problems.example.assets.functions import power_, fft_, ifft_, ifft_shift_, rfspectrum_, phase_

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof
from algorithms.parameter_optimization import parameters_optimize

from lib.functions import parse_command_line_args, custom_library
from config import config

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    # io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io = InputOutput(directory='noise_supression', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=False)
    # io.save_machine_metadata(io.save_path)

    custom_library(VariablePowerSplitter, DelayLine)

    ga_opts = {'n_generations': 12,
               'n_population': 4,
               'n_hof': 3,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=10e-9, n_samples=2**14, central_wl=1.55e-6)
    NoisySignal.protected = True
    ElectricFieldMeasurementDevice.protected = True

    noisy_signal = NoisySignal()
    md = ElectricFieldMeasurementDevice()

    true_signal = copy.deepcopy(noisy_signal.propagate(propagator.state, propagator))

    # evolver = HessianProbabilityEvolver(verbose=False)
    evolver = OperatorBasedProbEvolver(verbose=False)

    system = 1
    if system == 0:
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 1: VariablePowerSplitter(),
                 'sink': TerminalSink()}
        dl = DelayLine(parameters=[0])
        dl.node_lock = True
        edges = {('source', 0, 0): noisy_signal,
                 (0, 1, 0): dl,
                 (0, 1, 1): DelayLine(),
                 # (0, 1, 2): DispersiveFiber(),
                 # (0, 1, 3): DelayLine(),
                 (1, 'sink', 0): md,
                 }
    elif system == 1:
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 'sink': TerminalSink()}

        edges = {('source', 0, 0): noisy_signal,
                 (0, 'sink', 0): md,
                 }

    elif system == 2:
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 1: VariablePowerSplitter(),
                 'sink': TerminalSink()}

        edges = {('source', 0, 0): noisy_signal,
                 # (0, 1, 0): DispersiveFiber(parameters=[0]),
                 (0, 1, 0): WaveShaper(),
                 (1, 'sink', 0): md,
                 }

    elif system == 3:
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 1: VariablePowerSplitter(),
                 'sink': TerminalSink()}

        edges = {('source', 0, 0): noisy_signal,
                 (0, 1, 0): DispersiveFiber(parameters=[0]),
                 (1, 'sink', 0): md,
                 }

    evaluator = NoiseSuppression(propagator, true_signal=true_signal)

    graph = Graph.init_graph(nodes=nodes, edges=edges)

    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    graph.propagate(propagator)
    true_signal_noise = graph.measure_propagator(('source', 0, 0))

    # graph.sample_parameters()
    attr = graph.extract_attributes_to_list_experimental(attributes=['parameters', 'parameter_names'])
    update_rule = 'tournament'

    #%%
    optimize = 1
    if optimize == 0:
        loss = evaluator.evaluate_graph(graph, propagator)
        print(f"Loss function: {loss}")
        graph, x, score, log = parameters_optimize(graph, x0=None, method='L-BFGS+GA', verbose=True, log_callback=True)
    elif optimize == 1:
        hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                         ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                         # parameter_opt_method='CHEAP',
                                         parameter_opt_method='L-BFGS+GA',
                                         include_dashboard=False, crossover_maker=None,
                                         save_all_minimal_graph_data=True, save_all_minimal_hof_data=False)
        graph = hof[0][1]
        graph.draw()
    #%%

    noisy_signal.enable_noise()

    #%%

    graph.propagate(propagator)

    fig, axs = plt.subplots(2, 1)
    # state = graph.measure_propagator(('source', 0, 0))
    state = graph.measure_propagator('sink')

    kwargs = dict(alpha=0.5, lw=1,)
    axs[0].plot(propagator.t, power_(state), label='Measured Signal', **kwargs)
    axs[0].plot(propagator.t, power_(true_signal), label='True Signal', **kwargs)
    axs[0].plot(propagator.t, power_(evaluator.true_signal), label='Evaluator True Signal', **kwargs)
    axs[0].plot(propagator.t, power_(true_signal_noise), label='Input Signal', ls='--', **kwargs)

    axs[1].plot(propagator.f, np.abs(ifft_shift_(fft_(state, propagator.dt))), label='Measured Signal', **kwargs)
    axs[1].plot(propagator.f, np.abs(ifft_shift_(fft_(true_signal, propagator.dt))), label='True Signal', **kwargs)
    axs[1].plot(propagator.f, np.abs(ifft_shift_(fft_(evaluator.true_signal, propagator.dt))), label='Evaluator True Signal', **kwargs)
    axs[1].plot(propagator.f, np.abs(ifft_shift_(fft_(true_signal_noise, propagator.dt))), label='Input Signal', **kwargs)

    axs[0].set(xlabel='Time')
    axs[1].set(xlabel='Frequency')
    axs[0].legend()
    axs[1].legend()

    print(config.NODE_TYPES_ALL)

    #%%


    # # save_hof(hof, io)
    #
    # graph = hof[0][1]
    # print(f"evaluation score is {evaluator.evaluate_graph(graph, propagator)}")
    #
    # #%%
    # graph.draw()
    # sink_node = [node for node in graph.nodes if graph.get_out_degree(node) == 0][0]
    # phase_shifter = [graph.edges[edge]['model'] for edge in graph.edges if type(graph.edges[edge]['model']).__name__ == 'PhaseShifter'][0]

    # plot_hof(hof, propagator, evaluator, io)
    #
    # fig, ax = plt.subplots(1, 1, figsize=[5, 3])
    # ax.fill_between(log['generation'], log['best'], log['mean'], color='grey', alpha=0.2)
    # ax.plot(log['generation'], log['best'], label='Best')
    # ax.plot(log['generation'], log['mean'], label='Population mean')
    # ax.plot(log['generation'], log['minimum'], color='darkgrey', label='Population minimum')
    # ax.plot(log['generation'], log['maximum'], color='black', label='Population maximum')
    # ax.set(xlabel='Generation', ylabel='Cost')
    # ax.legend()
    #
    # io.save_fig(fig, 'topology_log.png')