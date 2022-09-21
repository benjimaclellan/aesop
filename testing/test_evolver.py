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

from lib.functions import InputOutput

from simulator.fiber.evolver import HessianProbabilityEvolver, OperatorBasedProbEvolver
from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evaluator_subclasses.evaluator_phase_sensitivity import PhaseSensitivity

from simulator.fiber.node_types_subclasses.inputs import ContinuousWaveLaser
from simulator.fiber.node_types_subclasses.outputs import MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import PhaseShifter
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from simulator.fiber.node_types_subclasses import *

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof

from lib.functions import parse_command_line_args

# np.random.seed(10)

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path='phase_sensitivity_test', unique_id=False)
    io.save_machine_metadata(io.save_path)

    PhaseShifter.protected = True
    ga_opts = {'n_generations': 12,
               'n_population': 6,
               'n_hof': 6,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=10e-9, n_samples=2 ** 14, central_wl=1.55e-6)

    phase, phase_node = (0.5 * np.pi, 'phase-shift')
    phase_shifter = PhaseShifter(parameters=[phase])
    phase_shifter.protected = True

    md = MeasurementDevice()
    md.protected = True

    # evolver = HessianProbabilityEvolver(verbose=False)
    evolver = OperatorBasedProbEvolver(verbose=False)

    for _i in range(10):
        nodes = {'source': TerminalSource(),
                 0: VariablePowerSplitter(),
                 1: VariablePowerSplitter(),
                 'sink': TerminalSink()}

        edges = {('source', 0): ContinuousWaveLaser(),
                 (0, 1, 0): phase_shifter,
                 (1, 'sink', 0): md,
                 }
        evaluator = PhaseSensitivity(propagator, phase=phase, phase_model=PhaseShifter())

        graph = Graph.init_graph(nodes=nodes, edges=edges)

        graph.assert_number_of_edges()
        graph.initialize_func_grad_hess(propagator, evaluator)

        stop = ''
        # while stop == '':
        for _j in range(100):
            graph_tmp = copy.deepcopy(graph)
            evolver.create_graph_matrix(graph, evaluator)
            graph, node_or_edge = evolver.evolve_graph(graph, evaluator)
            print(graph)
            phase_models = [graph.edges[edge]['model'] for edge in graph.edges if type(graph.edges[edge]['model']) == type(phase_shifter)]
            if len(phase_models) != 1:
                fig, axs = plt.subplots(1, 2)
                graph.draw(ax=axs[1])
                graph_tmp.draw(ax=axs[0])
                raise ValueError('not the right number of phase shifters')
            # stop = input()