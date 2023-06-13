"""
Discrete time lens
"""

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
import scipy.io
from scipy.constants import speed_of_light
import autograd.numpy as np
from config import config
from lib.functions import InputOutput

from simulator.fiber.evolver import HessianProbabilityEvolver, OperatorBasedProbEvolver
from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evaluator_subclasses.evaluator_discrete_time_lens import DiscreteTimeLens

from simulator.fiber.node_types_subclasses.inputs import Impulse, PulsedLaser, NoisySignal
from simulator.fiber.node_types_subclasses.outputs import ElectricFieldMeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import PhaseModulator, DispersiveFiber, OpticalAmplifier
from simulator.fiber.node_types_subclasses.single_path import SquareFilter

from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from simulator.fiber.assets.functions import power_, fft_, fft_shift_, ifft_, ifft_shift_, psd_, phase_, phase_spectrum_
from simulator.fiber.assets.additive_noise import AdditiveNoise

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof
from algorithms.parameter_optimization import parameters_optimize

from lib.functions import parse_command_line_args, custom_library
from config import config

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    # io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io = InputOutput(directory='time_lens_with_amplifier2', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=False)

    data = scipy.io.loadmat(r'/Users/benjamin/Library/CloudStorage/OneDrive-UniversityofWaterloo/Desktop/2 - Papers/2022 - MacLellan - Inverse design of photonic signal processing systems/experimental_comparison/forPiotrBen.mat')

    central_wl = speed_of_light/192.93e12
    propagator = Propagator(window_t=20e-9, n_samples=2**14, central_wl=central_wl)
    Impulse.protected = True
    ElectricFieldMeasurementDevice.protected = True
    AdditiveNoise.simulate_with_noise = False

    input_laser = PulsedLaser(parameters_from_name={'pulse_width': 1e-10, 'peak_power': 1.0,
                                                    't_rep': 1/80e6, 'pulse_shape': 'delta',
                                                    'central_wl': central_wl, 'train': False})
    input_laser.parameter_locks = input_laser.number_of_parameters * [True]
    amp = OpticalAmplifier()
    sf = SquareFilter(parameters_from_name={"bandwidth": 5e11})
    sf.parameter_locks = [True]
    # pm = PhaseModulator(parameters_from_name={'depth': 11, 'frequency': 5e9, 'shift': 0.0})
    pm = PhaseModulator(parameters_from_name={'depth': 11, 'frequency': 6.92969051e+09, 'shift': 0.0})
    # pm.parameter_locks = [False, True, True, True]
    df = DispersiveFiber(parameters_from_name={'length': 5e3})

    impulse = Impulse()
    md = ElectricFieldMeasurementDevice()

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             3: VariablePowerSplitter(),
             4: VariablePowerSplitter(),
             'sink': TerminalSink()}
    edges = {('source', 0): input_laser,
             (0, 1): amp,
             (1, 2): sf,
             (2, 3): df,
             (3, 4): pm,
             (4, 'sink'): md,
             }
    graph = Graph.init_graph(nodes=nodes, edges=edges)

    evaluator = DiscreteTimeLens(propagator)

    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    opts = {"population_size": 100, "n_generations": 30, "maxiter": 200}

    for repeat in range(30):
        graph = Graph.init_graph(nodes=nodes, edges=edges)
        graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

        graph, x, score, logger = parameters_optimize(graph, method='L-BFGS+GA',
                                                      verbose=True, log_callback=False, **opts)

        #%%
        # params, models, parameter_index, _, _ = graph.extract_parameters_to_list()
        # graph.distribute_parameters_from_list(x, models, parameter_index)

        # pm_params = graph.edges[(2, 3, 0)]["model"].parameters
        # pm_params[0] = 0.6
        # graph.edges[(2, 3, 0)]["model"].set_parameters(pm_params)

        graph.propagate(propagator)

        input =  np.abs(fft_shift_(fft_(graph.measure_propagator((0, 1, 0)), propagator.dt)))
        output = np.abs(fft_shift_(fft_(graph.measure_propagator((4, 'sink', 0)), propagator.dt)))

        fig, ax = plt.subplots()
        colors = ["salmon", "teal"]

        ax.plot(data["f__THz"], data["In"], color=colors[0], label="input (experiment)")
        ax.plot(data["f__THz"], data["Out"], color=colors[1], label="output (experiment)")

        ax.fill_between((propagator.f.squeeze() + propagator.central_frequency)/1e12, np.zeros_like(propagator.f.squeeze()),
                        evaluator.target_psd_osc.squeeze(), alpha=0.3)
        ax.fill_between((propagator.f.squeeze() + propagator.central_frequency)/1e12, np.zeros(propagator.n_samples),
                        input.squeeze()/np.max(input), color=colors[0], label="input (aesop)", alpha=0.5)
        ax.fill_between((propagator.f.squeeze() + propagator.central_frequency)/1e12, np.zeros(propagator.n_samples),
                        output.squeeze()/np.max(input), color=colors[1], label="output (aesop)", alpha=0.5)

        ax.set(xlabel="Frequency (THz)", ylabel="Intensity (arb. units)")
        ax.legend()
        fig.suptitle(f"run{repeat}")
        plt.show()

        score = evaluator.evaluate_graph(graph, propagator)
        print(score)
        io.save_fig(fig, filename=f"optimal_time_lens_{repeat}.png")
        #%%
        io.save_object(graph.duplicate_and_simplify_graph(graph), filename=f"optimal_time_lens_{repeat}.pkl")
        plt.pause(0.2)
