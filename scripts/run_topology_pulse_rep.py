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

import config.config as config

from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.evolver import Evolver, CrossoverMaker, StochMatrixEvolver, SizeAwareMatrixEvolver, ReinforcementMatrixEvolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.evolution_operators.evolution_operators import SwapNode

from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_pulserep import PulseRepetition

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization


plt.close('all')
if __name__ == '__main__':

    fixed_input = True
    if fixed_input:
        SwapNode.potential_node_types.remove('Input')

    io = InputOutput(directory='testing', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=True)
    io.save_machine_metadata(io.save_path)

    ga_opts = {'n_generations': 8,
               'n_population': 16, # psutil.cpu_count(),
               'n_hof': 2,
               'verbose': True,
               'num_cpus': psutil.cpu_count()}


    propagator = Propagator(window_t = 100e-9, n_samples = 2**14, central_wl=1.55e-6)

    pulse_width, rep_t, peak_power = (0.3e-9, 20.0e-9, 1.0)
    p, q = (1, 2)

    input_laser = PulsedLaser(parameters_from_name={'pulse_width':pulse_width, 'peak_power':peak_power, 't_rep':rep_t,
                                           'pulse_shape':'gaussian', 'central_wl':1.55e-6, 'train':True})
    input_laser.node_lock = True
    input = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)
    target = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width * (p / q), rep_t=rep_t * (p / q), peak_power=peak_power * (p / q))
    evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)

    evolver = StochMatrixEvolver(verbose=False)

    nodes = {0:input_laser,
             -1:MeasurementDevice()}
    edges = [(0,-1)]

    start_graph = Graph(nodes, edges, propagate_on_edges = False)
    start_graph.assert_number_of_edges()
    start_graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    update_rule = 'preferential'

    plt.plot(evaluator.target)
    plt.plot()

    graph, score, log = topology_optimization(copy.deepcopy(start_graph), propagator, evaluator, evolver, io,
                                              ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                              crossover_maker=None)
