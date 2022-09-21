"""
Test of topology optimization routines
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

from simulator.fiber.evolver import HessianProbabilityEvolver, OperatorBasedProbEvolver
from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evaluator_subclasses.evaluator_filter_design import FilterDesign

from simulator.fiber.node_types_subclasses.inputs import Impulse, PulsedLaser, NoisySignal
from simulator.fiber.node_types_subclasses.outputs import ElectricFieldMeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import PhaseModulator, IntegratedDelayLine, PhaseShifter, Waveguide, DispersiveFiber
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from simulator.fiber.assets.functions import power_, fft_, ifft_, ifft_shift_, rfspectrum_, phase_, phase_spectrum_
from simulator.fiber.assets.additive_noise import AdditiveNoise

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof
from algorithms.parameter_optimization import parameters_optimize

from lib.functions import parse_command_line_args, custom_library
from config import config

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    # io = InputOutput(directory='filter_design', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=True)
    io.save_machine_metadata(io.save_path)

    custom_library(VariablePowerSplitter, IntegratedDelayLine, PhaseShifter)

    ga_opts = {'n_generations': 55,
               'n_population': 10,
               'n_hof': 10,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=10e-9, n_samples=2**14, central_wl=1.55e-6)
    Impulse.protected = True
    ElectricFieldMeasurementDevice.protected = True
    AdditiveNoise.simulate_with_noise = False

    impulse = Impulse()
    md = ElectricFieldMeasurementDevice()

    true_impulse = copy.deepcopy(impulse.propagate(propagator.state, propagator))

    # evolver = HessianProbabilityEvolver(verbose=False)
    evolver = OperatorBasedProbEvolver(verbose=False)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             'sink': TerminalSink()}
    edges = {('source', 0, 0): impulse,
             (0, 1, 0): Waveguide(),
             (1, 'sink', 0): md,
             }
    evaluator = FilterDesign(propagator)

    graph = Graph.init_graph(nodes=nodes, edges=edges)

    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    graph.propagate(propagator)
    true_signal_noise = graph.measure_propagator(('source', 0, 0))

    # graph.sample_parameters()
    attr = graph.extract_attributes_to_list_experimental(attributes=['parameters', 'parameter_names'])
    update_rule = 'tournament'

    hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                     ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                     # parameter_opt_method='CHEAP',
                                     parameter_opt_method='L-BFGS+GA',
                                     include_dashboard=False, crossover_maker=None,
                                     save_all_minimal_graph_data=True, save_all_minimal_hof_data=True)
    graph = hof[0][1]
    graph.draw()
    attr = graph.extract_attributes_to_list_experimental(attributes=['parameters', 'parameter_names'])
    print(attr['parameters'])

    #%%

    graph.propagate(propagator)

    fig, axs = plt.subplots(ga_opts['n_hof'], 3, figsize=[10, ga_opts['n_hof']*2])

    kwargs = dict(alpha=1.0, lw=1,)
    for i, (score, graph) in enumerate(hof):
        graph.propagate(propagator)
        state = graph.measure_propagator('sink')
        graph.draw(axs[i, 0])

        axs[i, 1].plot(propagator.wl/1e-6, np.abs(ifft_shift_(fft_(state, propagator.dt))), label='Measured Signal', **kwargs)
        # axs[i, 0].plot(propagator.wl/1e-6, np.abs(ifft_shift_(fft_(true_impulse, propagator.dt))), label='True Signal', **kwargs)
        axs[i, 1].plot(propagator.wl/1e-6, evaluator.target_transfer, label='Evaluator True Signal', **kwargs)

        axs[i, 2].plot(propagator.wl/1e-6, np.log10(np.abs(ifft_shift_(fft_(state, propagator.dt))))*10, label='Measured Signal', **kwargs)
        # axs[i, 1].plot(propagator.wl/1e-6, np.log10(np.abs(ifft_shift_(fft_(true_impulse, propagator.dt))))*10, label='True Signal', **kwargs)
        axs[i, 2].plot(propagator.wl/1e-6, np.log10(evaluator.target_transfer)*10, label='Evaluator True Signal', **kwargs)

    io.save_fig(fig, 'hof.png')

    #%%
    fig, ax = plt.subplots(1, 1)
    ax.plot(log['time'], log['best'])
    io.save_fig(fig, 'log.png')
