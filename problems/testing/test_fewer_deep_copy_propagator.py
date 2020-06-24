import pytest
import autograd.numpy as np
from autograd import grad
import numpy.testing as testing
import matplotlib.pyplot as plt

from algorithms.parameter_optimization_utils import get_individual_score, get_initial_population, adam_function_wrapper

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import power_, psd_

from lib.analysis.hessian import function_wrapper


GRAPHICAL_TESTING = False

# ---------------------------- Providers --------------------------------
def get_graph(deep_copy):
    """
    Returns the default graph for testing, with fixed topology at this time
    """
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1, 'central_wl': 1.55e-6}),
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: WaveShaper(),
             3: DelayLine(),
             4: MeasurementDevice()
             }
    edges = [(0,1, CorningFiber(parameters=[0])),
             (1,2, CorningFiber(parameters=[0])),
             (2,3),
             (3,4)]

    graph = Graph(nodes, edges, propagate_on_edges=True, deep_copy=deep_copy)
    graph.assert_number_of_edges()
    return graph


def get_propagator():
    return Propagator(window_t = 1e-8, n_samples = 2**14, central_wl=1.55e-6)


def get_evaluator(propagator):
    return RadioFrequencyWaveformGeneration(propagator)


def param_pool():
    graph = get_graph(False)
    propagator = get_propagator()
    evaluator = get_evaluator(propagator)
    pop, _, _ = get_initial_population(graph, propagator, evaluator, 32, 'uniform')
    
    return [param for (score, param) in pop]

param_list = param_pool()


def display_states_deepcopy_noDeepcopy(state_deepcopy, state_no_deepcopy, propagator):
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(propagator.t, power_(state_deepcopy), label='deepcopy')
    ax[0].plot(propagator.t, power_(state_no_deepcopy), label='not deepcopy', linestyle=':')

    ax[1].plot(propagator.f, psd_(state_deepcopy, propagator.dt, propagator.df), label='deepcopy')
    ax[1].plot(propagator.f, psd_(state_no_deepcopy, propagator.dt, propagator.df), label='not deepcopy', linestyle=':')

    ax[0].legend()
    ax[1].legend()
    plt.show()

@pytest.fixture(scope='function')
def graph_deepcopy():
    return get_graph(True)


@pytest.fixture(scope='function')
def graph_no_deepcopy():
    return get_graph(False)


@pytest.fixture(scope='function')
def propagator():
    return get_propagator()


@pytest.fixture(scope='function')
def evaluator(propagator):
    return get_evaluator(propagator)


@pytest.fixture(scope='function')
def nodeEdgeIndex_parameterIndex(graph_no_deepcopy):
    _, node_edge_index, parameter_index, _, _ = graph_no_deepcopy.extract_parameters_to_list()
    return node_edge_index, parameter_index


@pytest.mark.parametrize("params", param_list)
def test_evaluation_equality(params, graph_deepcopy, graph_no_deepcopy, propagator, evaluator, nodeEdgeIndex_parameterIndex):
    assert get_individual_score(graph_deepcopy, propagator, evaluator,
                                params, nodeEdgeIndex_parameterIndex[0],
                                nodeEdgeIndex_parameterIndex[1]) == \
           get_individual_score(graph_no_deepcopy, propagator, evaluator,
                                params, nodeEdgeIndex_parameterIndex[0],
                                nodeEdgeIndex_parameterIndex[1])

@pytest.mark.xfail
@pytest.mark.parametrize("params", param_list)
def test_gradient_equality(params, graph_deepcopy, graph_no_deepcopy, propagator, evaluator, nodeEdgeIndex_parameterIndex):
    # setup the gradient function and bounds
    fitness_funct_deepcopy = function_wrapper(graph_deepcopy, propagator, evaluator, exclude_locked=True)
    adam_fitness_funct_deepcopy = adam_function_wrapper(fitness_funct_deepcopy)
    fitness_grad_deepcopy = grad(adam_fitness_funct_deepcopy)
    
    fitness_funct_no_deepcopy = function_wrapper(graph_no_deepcopy, propagator, evaluator, exclude_locked=True)
    adam_fitness_funct_no_deepcopy = adam_function_wrapper(fitness_funct_no_deepcopy)
    fitness_grad_no_deepcopy = grad(adam_fitness_funct_no_deepcopy)

    assert fitness_grad_deepcopy(params, 0) == fitness_grad_no_deepcopy(params, 0)

@pytest.mark.parametrize("params", param_list)
def test_propagation_result_equality(params, graph_deepcopy, graph_no_deepcopy, propagator, evaluator, nodeEdgeIndex_parameterIndex):
    propagator_nodeepcopy = get_propagator()

    # deepcopy
    graph_deepcopy.distribute_parameters_from_list(params, nodeEdgeIndex_parameterIndex[0], nodeEdgeIndex_parameterIndex[1])
    graph_deepcopy.propagate(propagator)
    deepcopy_state = graph_deepcopy.nodes[len(graph_deepcopy.nodes) - 1]['states'][0]

    # not deepcopy
    graph_no_deepcopy.distribute_parameters_from_list(params, nodeEdgeIndex_parameterIndex[0], nodeEdgeIndex_parameterIndex[1])
    graph_no_deepcopy.propagate(propagator_nodeepcopy)
    no_deepcopy_state = graph_no_deepcopy.nodes[len(graph_no_deepcopy.nodes) - 1]['states'][0]

    if GRAPHICAL_TESTING:
        display_states_deepcopy_noDeepcopy(deepcopy_state, no_deepcopy_state, propagator)

    testing.assert_allclose(deepcopy_state, no_deepcopy_state)