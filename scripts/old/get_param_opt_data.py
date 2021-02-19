
# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# various imports
import copy
import autograd.numpy as np
import random

from lib.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import Photodiode
from problems.example.node_types_subclasses.single_path import WaveShaper, IntensityModulator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types import TerminalSource, TerminalSink

from algorithms.functions import ParameterOptimizationLogger
from algorithms.parameter_optimization import parameters_optimize

# saving data
from lib.functions import InputOutput

def handle_io():
    io = InputOutput(directory='testing', verbose=True)
    io.init_save_dir(sub_path=None, unique_id=True)
    io.save_machine_metadata(io.save_path)
    return io

def test_graph0():
    propagator = Propagator(window_t=1e-9, n_samples=2 ** 14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    nodes = {'source':TerminalSource(),
            0:VariablePowerSplitter(),
            1:VariablePowerSplitter(),
            2:VariablePowerSplitter(),
            'sink':TerminalSink()
            }
    edges = {('source', 0):ContinuousWaveLaser(),
             (0, 1): IntensityModulator(),
             (1, 2): WaveShaper(), 
             (2,'sink'):Photodiode(),
            }
    graph = Graph.init_graph(nodes=nodes, edges=edges)
    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator)
    return graph, propagator, evaluator

def compare_convergence_by_algorithm(save=True, show=True):
    """
    Compares convergence of random algorithms, with same starting point to each
    """
    graph, propagator, evaluator = test_graph0()
    
    if save:
        io = handle_io()
        io.save_object(object_to_save=graph, filename=f"start_graph.pkl")
        io.save_object(object_to_save=propagator, filename=f"propagator.pkl")
        io.save_object(object_to_save=evaluator, filename=f"evaluator.pkl")

    x0 = copy.deepcopy(graph.sample_parameters_to_list())
    _, models, param_index, *_ = graph.extract_parameters_to_list()

    logger_dict = {}

    for alg in ['L-BFGS', 'GA', 'ADAM', 'ADAM+GA', 'L-BFGS+GA', 'PSO', 'L-BFGS+PSO']:
        graph.distribute_parameters_from_list(copy.deepcopy(x0), models, param_index)
        _, params, _, logger = parameters_optimize(graph, method=alg, log_callback=True)
        logger_dict[alg] = logger
        fig, _ = logger.display_log(show=show)
        if save:
            io.save_fig(fig=fig, filename=f'{alg}.png')
            io.save_object(object_to_save=params, filename=f'{alg}_graph_params.pkl')
            io.save_object(object_to_save=logger.get_log(), filename=f'{alg}_dataframe.pkl')

    fig, _ = ParameterOptimizationLogger.display_log_comparison(logger_dict, title='Comparing algorithm convergence', show=show)
    if save:
       io.save_fig(fig=fig, filename='comparing_alg_convergence.png')
        
def compare_convergence_by_starting_point(save=True, show=True):
    """
    Compares the convergence of the same algorithm from different startpoints
    """
    graph, propagator, evaluator = test_graph0()

    if save:
        io = handle_io()
        io.save_object(object_to_save=graph, filename=f"start_graph.pkl")
        io.save_object(object_to_save=propagator, filename=f"propagator.pkl")
        io.save_object(object_to_save=evaluator, filename=f"evaluator.pkl")

    num_startpoints = 7
    logger_dict = {}

    for alg in ['L-BFGS', 'GA', 'ADAM', 'ADAM+GA', 'L-BFGS+GA', 'PSO', 'L-BFGS+PSO']:
        for i in range(num_startpoints):
            graph.sample_parameters()
            _, params, _, logger = parameters_optimize(graph, method=alg, log_callback=True)
            logger_dict[f'start point {i}'] = logger

            if save:
                io.save_object(object_to_save=params, filename=f'{alg}{i}_graph_params.pkl')
                io.save_object(object_to_save=logger.get_log(), filename=f'{alg}{i}_dataframe.pkl')
        
        fig, _ = ParameterOptimizationLogger.display_log_comparison(logger_dict, title=f'{alg}: convergence from different start points')

        if save:       
            io.save_fig(fig=fig, filename=f'{alg}_convergence_from_diff_starts.png')


if __name__ == "__main__":
    random.seed(3030)
    np.random.seed(204544)
    # compare_convergence_by_algorithm()
    compare_convergence_by_starting_point()