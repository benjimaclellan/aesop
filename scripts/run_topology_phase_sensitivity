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
import numpy as np

import config.config as config

from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.evolver import Evolver, CrossoverMaker, StochMatrixEvolver, SizeAwareMatrixEvolver, ReinforcementMatrixEvolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.evolution_operators.evolution_operators import SwapNode

from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.evaluator_subclasses.evaluator_pulserep import PulseRepetition
from problems.example.evaluator_subclasses.evaluator_phase_sensitivity import PhaseSensitivity

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.single_path import DelayLine, IntensityModulator, ProgrammableFilter, EDFA
from problems.example.node_types_subclasses.single_path import PhaseShifter
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization


plt.close('all')
if __name__ == '__main__':

    io = InputOutput(directory='testing', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=True)
    io.save_machine_metadata(io.save_path)

    ga_opts = {'n_generations': 4,
               'n_population': 4, # psutil.cpu_count(),
               'n_hof': 6,
               'verbose': True,
               'num_cpus': psutil.cpu_count()}

    propagator = Propagator(window_t=100e-9, n_samples=2 ** 14, central_wl=1.55e-6)
    evolver = Evolver()

    phase, phase_node = (0.5 * np.pi, -2)
    phase_shifter = PhaseShifter(parameters=[phase])
    evaluator = PhaseSensitivity(propagator, phase=phase, phase_node=phase_node)

    nodes = {0: ContinuousWaveLaser(),
             phase_node: phase_shifter,
             -1: Photodiode(parameters_from_name={'bandwidth': 1 / propagator.window_t})}
    edges = [(0, phase_node), (phase_node, -1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    start_graph = copy.deepcopy(graph)
    def __evolve_graph(_graph, _evaluator):
        _new_graph, _evo_op = Evolver().evolve_graph(_graph, _evaluator)
        flag = False
        for node in _graph.nodes:
            if node == phase_node:
                flag = True
        if flag:
            return _new_graph, _evo_op
        else:
            print('need to start again from start_graph, as PS has been removed')
            return copy.deepcopy(start_graph), None

    evolver.evolve_graph = __evolve_graph

    update_rule = 'preferential'

    # for j in range(10):
    #     fig, ax = plt.subplots(1,1)
    #     for i in range(10):
    #         graph, evo_op = evolver.evolve_graph(graph, evaluator)
    #         ax.cla()
    #         graph.draw(ax=ax)
    #         plt.waitforbuttonpress()

    graph, score, log = topology_optimization(copy.deepcopy(start_graph), propagator, evaluator, evolver, io,
                                              ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                              crossover_maker=None)
