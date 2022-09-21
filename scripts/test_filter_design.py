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

    # io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io = InputOutput(directory='filter_design', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=False)
    # io.save_machine_metadata(io.save_path)

    custom_library(VariablePowerSplitter, IntegratedDelayLine, PhaseShifter)

    ga_opts = {'n_generations': 12,
               'n_population': 4,
               'n_hof': 5,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=10e-9, n_samples=2**14, central_wl=1.55e-6)
    Impulse.protected = True
    ElectricFieldMeasurementDevice.protected = True
    AdditiveNoise.simulate_with_noise = False

    impulse = Impulse()
    # impulse = NoisySignal(parameters_from_name={'pulse_shape': 'sinc', 'pulse_width': 1e-10})
    md = ElectricFieldMeasurementDevice()

    true_impulse = copy.deepcopy(impulse.propagate(propagator.state, propagator))

    # evolver = HessianProbabilityEvolver(verbose=False)
    evolver = OperatorBasedProbEvolver(verbose=False)
    # VariablePowerSplitter.node_lock = True

    system = 0
    if system == 0:
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 1: VariablePowerSplitter(),
                 'sink': TerminalSink()}
        edges = {('source', 0, 0): impulse,
                 (0, 1, 0): Waveguide(),
                 (1, 'sink', 0): md,
                 }
    elif system == 1:
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 1: VariablePowerSplitter(),
                 2: VariablePowerSplitter(),
                 3: VariablePowerSplitter(),
                 4: VariablePowerSplitter(),
                 5: VariablePowerSplitter(),
                 6: VariablePowerSplitter(),
                 'sink': TerminalSink()}
        id = IntegratedDelayLine(parameters=[0.0])
        id.node_lock = True
        mdelay = 0.0
        ph = 0.0
        edges = {('source', 0, 0): impulse,
                 (0, 1, 0): IntegratedDelayLine(parameters=[mdelay]),
                 (1, 2, 0): IntegratedDelayLine(parameters=[mdelay]),
                 (2, 3, 0): IntegratedDelayLine(parameters=[mdelay]),
                 (3, 4, 0): IntegratedDelayLine(parameters=[mdelay]),
                 (4, 5, 0): IntegratedDelayLine(parameters=[mdelay]),
                 (5, 6, 0): IntegratedDelayLine(parameters=[mdelay]),
                 (0, 1, 1): PhaseShifter(parameters=[ph]),
                 (1, 2, 1): PhaseShifter(parameters=[ph]),
                 (2, 3, 1): PhaseShifter(parameters=[ph]),
                 (3, 4, 1): PhaseShifter(parameters=[ph]),
                 (4, 5, 1): PhaseShifter(parameters=[ph]),
                 (5, 6, 1): PhaseShifter(parameters=[ph]),
                 # (0, 1, 1): id,
                 # (1, 2, 1): id,
                 # (2, 3, 1): id,
                 # (3, 4, 1): id,
                 # (4, 5, 1): id,
                 # (5, 6, 1): id,
                 # (0, 1, 1): IntegratedDelayLine(parameters=[0.0]),
                 # (1, 2, 1): IntegratedDelayLine(parameters=[0.0]),
                 # (2, 3, 1): IntegratedDelayLine(parameters=[0.0]),
                 # (3, 4, 1): IntegratedDelayLine(parameters=[0.0]),
                 # (4, 5, 1): IntegratedDelayLine(parameters=[0.0]),
                 (6, 'sink', 0): md,
                 }
    elif system == 14:
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 1: VariablePowerSplitter(),
                 2: VariablePowerSplitter(),
                 # 3: VariablePowerSplitter(),
                 4: VariablePowerSplitter(),
                 'sink': TerminalSink()}
        edges = {('source', 0, 0): impulse,
                 (0, 1, 0): IntegratedDelayLine(parameters=[0.0]),
                 (0, 2, 1): IntegratedDelayLine(parameters=[0.0]),
                 (0, 4, 0): IntegratedDelayLine(parameters=[6.0]),
                 (1, 4, 0): IntegratedDelayLine(parameters=[1.0]),
                 (1, 4, 1): IntegratedDelayLine(parameters=[2.0]),
                 (2, 4, 0): IntegratedDelayLine(parameters=[4.0]),
                 (2, 4, 1): IntegratedDelayLine(parameters=[8.0]),
                 # (3, 4, 1): IntegratedDelayLine(parameters=[12]),
                 # (3, 4, 1): IntegratedDelayLine(parameters=[12]),
                 (4, 'sink', 0): md,
                 }
    elif system == 100:
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 1: VariablePowerSplitter(),
                 2: VariablePowerSplitter(),
                 3: VariablePowerSplitter(),
                 4: VariablePowerSplitter(),
                 'sink': TerminalSink()}
        edges = {('source', 0, 0): impulse,
                 (0, 1, 0): IntegratedDelayLine(parameters=[1.0]),
                 (0, 1, 1): IntegratedDelayLine(parameters=[0.0]),
                 (1, 2, 4): IntegratedDelayLine(parameters=[1.0]),
                 (1, 2, 4): IntegratedDelayLine(parameters=[1.0]),
                 (1, 3, 4): IntegratedDelayLine(parameters=[2.0]),
                 (1, 3, 4): IntegratedDelayLine(parameters=[25]),
                 # (3, 4, 1): IntegratedDelayLine(parameters=[12]),
                 # (3, 4, 1): IntegratedDelayLine(parameters=[12]),
                 (4, 'sink', 0): md,
                 }
    elif system == 2:
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 1: VariablePowerSplitter(),
                 'sink': TerminalSink()}
        edges = {('source', 0, 0): impulse,
                 (0, 1, 0):IntegratedDelayLine(parameters=[0]),
                 (0, 1, 1): IntegratedDelayLine(parameters=[5.1]),
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
                                         save_all_minimal_graph_data=True, save_all_minimal_hof_data=True)
        graph = hof[0][1]
        graph.draw()
        attr = graph.extract_attributes_to_list_experimental(attributes=['parameters', 'parameter_names'])
        print(attr['parameters'])
    if optimize == 2:
        loss = evaluator.evaluate_graph(graph, propagator)
        print(f"Loss function: {loss}")
        print(graph)
        graph.draw()
        print(attr['parameters'])

    #%%

    graph.propagate(propagator)

    fig, axs = plt.subplots(2, 1)
    # state = graph.measure_propagator(('source', 0, 0))
    state = graph.measure_propagator('sink')

    kwargs = dict(alpha=1.0, lw=1,)
    # axs[0].plot(propagator.t, power_(state), label='Measured Signal', **kwargs)
    # axs[0].plot(propagator.t, power_(true_impulse), label='True Signal', **kwargs)
    # # axs[0].plot(propagator.t, power_(evaluator.true_signal), label='Evaluator True Signal', **kwargs)

    # axs[1].plot(propagator.t, phase_(state), label='Measured Signal', **kwargs)
    # axs[1].plot(propagator.t, phase_(true_impulse), label='True Signal', **kwargs)
    # # axs[1].plot(propagator.t, phase_(evaluator.true_signal), label='Evaluator True Signal', **kwargs)

    axs[0].plot(propagator.wl/1e-6, np.abs(ifft_shift_(fft_(state, propagator.dt))), label='Measured Signal', **kwargs)
    axs[0].plot(propagator.wl/1e-6, np.abs(ifft_shift_(fft_(true_impulse, propagator.dt))), label='True Signal', **kwargs)
    axs[0].plot(propagator.wl/1e-6, evaluator.target_transfer, label='Evaluator True Signal', **kwargs)

    axs[1].plot(propagator.wl/1e-6, np.log10(np.abs(ifft_shift_(fft_(state, propagator.dt))))*10, label='Measured Signal', **kwargs)
    axs[1].plot(propagator.wl/1e-6, np.log10(np.abs(ifft_shift_(fft_(true_impulse, propagator.dt))))*10, label='True Signal', **kwargs)
    axs[1].plot(propagator.wl/1e-6, np.log10(evaluator.target_transfer)*10, label='Evaluator True Signal', **kwargs)

    # axs[1].set(xlabel='Time')
    axs[0].set(xlabel='Wavelen')
    axs[1].set(xlabel='Frequency')
    axs[0].legend()
    # axs[1].legend()

    print(config.NODE_TYPES_ALL)

    #%%

