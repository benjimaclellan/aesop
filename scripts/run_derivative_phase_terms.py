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
import time

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

from simulator.fiber.evaluator_subclasses.evaluator_phase_sensitivity import SecondOrderDifferentialSensitivity

from simulator.fiber.node_types_subclasses.inputs import ContinuousWaveLaser
from simulator.fiber.node_types_subclasses.outputs import Photodiode, MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import PhaseModulator, WaveShaper, \
    OpticalAmplifier, PhaseShifter, DispersiveFiber, IntegratedDelayLine, VariableOpticalAttenuator, IntensityModulator
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink


from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof
from algorithms.parameter_optimization import parameters_optimize

from lib.functions import parse_command_line_args, custom_library

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    # io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io = InputOutput(directory='20210721_second_order_derivative_terms', verbose=True)
    io.init_save_dir(sub_path='second_order_derivative_terms', unique_id=True)

    custom_library(VariablePowerSplitter, IntegratedDelayLine, PhaseShifter, VariableOpticalAttenuator,
                   OpticalAmplifier, DispersiveFiber, PhaseModulator, IntensityModulator)

    ga_opts = {'n_generations': 10,
               'n_population': 10,
               'n_hof': 3,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=10/12e9, n_samples=2**14, central_wl=1.55e-6)
    PhaseShifter.protected = True

    phase1, phase_node1 = (0.0 * np.pi, 'phase-shift1')
    phase_shifter1 = PhaseShifter(parameters=[phase1])
    phase_shifter1.protected = True

    phase2, phase_node2 = (0.0 * np.pi, 'phase-shift2')
    phase_shifter2 = PhaseShifter(parameters=[phase2])
    phase_shifter2.protected = True

    cw = ContinuousWaveLaser()
    cw.protected = True
    cw.node_lock = True

    md = MeasurementDevice()
    md.protected = True

    # evolver = HessianProbabilityEvolver(verbose=False)
    evolver = OperatorBasedProbEvolver(verbose=False)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             # 2: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): ContinuousWaveLaser(parameters=[0.05]),
             (0, 1, 0): phase_shifter1,
             (0, 1, 1): phase_shifter2,
             # (0, 1, 1): DispersiveFiber(parameters=[1]),
             (1, 'sink', 1): md,
             }
    evaluator = SecondOrderDifferentialSensitivity(propagator, phase1=phase1, phase2=phase2,
                                                   phase_model1=phase_shifter1, phase_model2=phase_shifter2)

    graph = Graph.init_graph(nodes=nodes, edges=edges)

    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    update_rule = 'tournament'

    graph_init = copy.deepcopy(graph)
    graph_opt = None
    loss = 0.0

    M, N = 6, 6
    for _i in range(M):
        graph = copy.deepcopy(graph_init)
        for _j in range(N):
            (graph, _) = evolver.evolve_graph(graph, evaluator=evaluator, generation=0)
            res = parameters_optimize(graph, x0=None, method='L-BFGS+GA', verbose=True)
            loss_curr = res[2]
            if loss_curr < loss:
                graph_opt = copy.deepcopy(graph)
                loss = loss_curr

    print(graph_opt)

    # print(evaluator.evaluate_graph(graph, propagator=propagator))
    # print(evaluator.evaluate_graph(graph, propagator=propagator))


    #%%

    # parameters_optimize(graph, x0=None, method='L-BFGS', verbose=True)

    # fig, ax = plt.subplots(1, 1)
    # graph.draw(ax=ax)
    # for i in range(0, 24):
    #     plt.cla()
    #     (graph, _) = evolver.evolve_graph(graph, evaluator=evaluator, generation=0)
    #     print(graph)
    #     graph.draw(ax=ax)
    #     evaluator.evaluate_graph(graph, propagator=propagator)
    #     plt.pause(0.25)


    # t0 = time.time()
    # hof, log = topology_optimization(copy.deepcopy(graph), propagator, copy.deepcopy(evaluator), evolver, io,
    #                                  ga_opts=ga_opts, local_mode=True, update_rule=update_rule,
    #                                  parameter_opt_method='CHEAP',
    #                                  # parameter_opt_method='L-BFGS+GA',
    #                                  include_dashboard=False, crossover_maker=None,
    #                                  save_all_minimal_graph_data=False, save_all_minimal_hof_data=False)
    # t1 = time.time()
    # save_hof(hof, io)
    #
    # io.save_machine_metadata(io.save_path, time=t1-t0)
    #
    # graph = hof[0][1]
    # print(f"evaluation score is {evaluator.evaluate_graph(graph, propagator)}")
    #
    # #%%
    # graph.draw()
    # sink_node = [node for node in graph.nodes if graph.get_out_degree(node) == 0][0]
    # phase_shifter1 = [graph.edges[edge]['model'] for edge in graph.edges if type(graph.edges[edge]['model']).__name__ == 'PhaseShifter'][0]
    #
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