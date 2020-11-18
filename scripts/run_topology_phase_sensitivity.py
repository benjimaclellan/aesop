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
import networkx as nx
import numpy as np
import types
import ray
from functools import wraps

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
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.single_path import DelayLine, IntensityModulator, ProgrammableFilter, OpticalAmplifier
from problems.example.node_types_subclasses.single_path import PhaseShifter
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof


plt.close('all')
if __name__ == '__main__':
    #
    io = InputOutput(directory='testing', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=True)
    io.save_machine_metadata(io.save_path)

    ga_opts = {'n_generations': 6,
               'n_population': 4, # psutil.cpu_count(),
               'n_hof': 2,
               'verbose': True,
               'num_cpus': psutil.cpu_count()}

    propagator = Propagator(window_t=10e-9, n_samples=2 ** 14, central_wl=1.55e-6)

    phase, phase_node = (0.5 * np.pi, -2)
    phase_shifter = PhaseShifter(parameters=[phase])
    evaluator = PhaseSensitivity(propagator, phase=phase, phase_node=phase_node)
    evolver = StochMatrixEvolver(verbose=False, permanent_nodes=[phase_node])

    def create_start_graph():

        nodes = {0: ContinuousWaveLaser(),
                 phase_node: phase_shifter,
                 -1: MeasurementDevice()}
        edges = [(0, phase_node), (phase_node, -1)]

        graph = Graph(nodes, edges, propagate_on_edges = False)
        graph.assert_number_of_edges()
        graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
        return graph

    graph = create_start_graph()

    # def __decorated_evolve_graph(base_evolve_graph_function):
    #     @wraps(base_evolve_graph_function)
    #     def assert_phase_shifter(*args, **kwargs):
    #         _evaluator = args[1]
    #         try:
    #             _new_graph, _evo_op = base_evolve_graph_function(*args, **kwargs)
    #             flag = False
    #             for node in _new_graph.nodes:
    #                 if node == phase_node:
    #                     flag = True
    #             if not flag:
    #                 raise RuntimeError
    #             return _new_graph, _evo_op
    #         except RuntimeError as e:
    #             return create_start_graph(), None
    #         return graph, evo_op
    #     return assert_phase_shifter
    #
    # evolver.evolve_graph = __decorated_evolve_graph(evolver.evolve_graph)
    # graph, evo_op = evolver.evolve_graph(graph, evaluator)

    # update_rule = 'preferential'
    update_rule = 'random'

    #%%
    for j in range(10):
        # fig, axs = plt.subplots(1,2)
        for i in range(1):
            graph, evo_op = evolver.evolve_graph(graph, evaluator)
            # ax = axs[0]
            # ax.cla()
            # graph.draw(ax=ax, method='kamada_kawai')
            #
            # ax = axs[1]
            # ax.cla()
            # nx.k_core(graph).draw(ax=axs[1], method='kamada_kawai')
            # # plt.pause(0.01)
            # plt.waitforbuttonpress()

    #%%
    # io.save_object(graph, 'test_graph.pkl')

    # hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
    #                                  ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
    #                                  include_dashboard=False, crossover_maker=None)
    #
    # save_hof(hof, io)
    # plot_hof(hof, propagator, evaluator, io)
