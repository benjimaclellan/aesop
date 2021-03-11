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

from problems.example.evolver import HessianProbabilityEvolver, OperatorBasedProbEvolver
from lib.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.evaluator_subclasses.evaluator_phase_sensitivity import PhaseSensitivity

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import Photodiode, MeasurementDevice
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, OpticalAmplifier, PhaseShifter, DispersiveFiber
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink


from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof

from lib.functions import parse_command_line_args

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    # io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io = InputOutput(directory='phase_sensitivity', verbose=True)
    io.init_save_dir(sub_path='phase_sensitivity', unique_id=False)
    io.save_machine_metadata(io.save_path)

    ga_opts = {'n_generations': 6,
               'n_population': 6,
               'n_hof': 3,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=10/12e9, n_samples=2**14, central_wl=1.55e-6)
    PhaseShifter.protected = True

    phase, phase_node = (0.0 * np.pi, 'phase-shift')
    phase_shifter = PhaseShifter(parameters=[phase])
    phase_shifter.protected = True

    cw = ContinuousWaveLaser()
    cw.protected = True

    md = MeasurementDevice()
    md.protected = True

    # evolver = HessianProbabilityEvolver(verbose=False)
    evolver = OperatorBasedProbEvolver(verbose=False)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): ContinuousWaveLaser(parameters=[0.005]),
             (0, 1, 0): phase_shifter,
             # (0, 1, 1): DispersiveFiber(parameters=[1]),
             (1, 'sink', 1): md,
             }
    evaluator = PhaseSensitivity(propagator, phase=phase, phase_model=phase_shifter)

    graph = Graph.init_graph(nodes=nodes, edges=edges)

    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    update_rule = 'tournament'
    #%%

    hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                     ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                     # parameter_opt_method='CHEAP',
                                     parameter_opt_method='L-BFGS+GA',
                                     include_dashboard=False, crossover_maker=None,
                                     save_all_minimal_graph_data=True, save_all_minimal_hof_data=True)
    save_hof(hof, io)

    graph = hof[0][1]
    print(f"evaluation score is {evaluator.evaluate_graph(graph, propagator)}")

    #%%
    graph.draw()
    sink_node = [node for node in graph.nodes if graph.get_out_degree(node) == 0][0]
    phase_shifter = [graph.edges[edge]['model'] for edge in graph.edges if type(graph.edges[edge]['model']).__name__ == 'PhaseShifter'][0]

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