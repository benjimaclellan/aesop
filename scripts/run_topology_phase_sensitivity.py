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
from problems.example.evolver import ProbabilityLookupEvolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator

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
from problems.example.node_types import TerminalSource, TerminalSink

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof


plt.close('all')
if __name__ == '__main__':

    io = InputOutput(directory='testing', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=True)
    io.save_machine_metadata(io.save_path)

    PhaseShifter.protected = True
    ga_opts = {'n_generations': 10,
               'n_population': 10,
               'n_hof': 2,
               'verbose': True,
               'num_cpus': psutil.cpu_count()}

    propagator = Propagator(window_t=10e-9, n_samples=2 ** 14, central_wl=1.55e-6)

    phase, phase_node = (0.5 * np.pi, 'phase-shift')
    phase_shifter = PhaseShifter(parameters=[phase])
    evolver = ProbabilityLookupEvolver(verbose=False)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): ContinuousWaveLaser(),
             (0, 1, 0): phase_shifter,
             (1, 'sink'): MeasurementDevice(),
             }
    evaluator = PhaseSensitivity(propagator, phase=phase, phase_model=PhaseShifter)

    graph = Graph.init_graph(nodes=nodes, edges=edges)

    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator)

    # update_rule = 'preferential'
    update_rule = 'random'

    #%%

    io.save_object(graph.duplicate_and_simplify_graph(graph), 'test_graph.pkl')

    test_evolution = False
    if test_evolution:
        for j in range(1):
            fig, ax = plt.subplots(1, 1)
            for i in range(3):
                graph, evo_op = evolver.evolve_graph(graph, evaluator)
                ax.cla()
                graph.draw(ax=ax, debug=False)
                # plt.waitforbuttonpress()
                print(evaluator.evaluate_graph(graph, propagator))
                plt.pause(0.1)
        io.save_object(graph.duplicate_and_simplify_graph(graph), 'test_graph.pkl')

    #%%
    io.save_object(graph, 'test_graph.pkl')

    hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                     ga_opts=ga_opts, local_mode=True, update_rule=update_rule,
                                     include_dashboard=False, crossover_maker=None)

    save_hof(hof, io)
    plot_hof(hof, propagator, evaluator, io)
