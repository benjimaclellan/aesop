import pytest
import autograd.numpy as np
from autograd import grad, hessian
import numpy.testing as testing
import matplotlib.pyplot as plt
import copy

from algorithms.parameter_optimization_utils import get_individual_score, get_initial_population, adam_function_wrapper

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import power_, psd_
from problems.example.evaluator import Evaluator


from lib.analysis.hessian import function_wrapper


GRAPHICAL_TESTING = False

"""
Summary of findings so far: we can replace the deepcopies in the nested for loops by not deepcopies, and it works IF we don't propagate on edges
                            If we do propagate on edges, there' more to it s
- VJP of np.copy is not supported
- The first deepcopy results in a derivative. The subsequent ones do not
- You cannot trick the system with recursion, alas

TODO (to try):
- using the builtin autograd list type for temp state?
"""

# ---------------------------- Providers --------------------------------
def get_graph(deep_copy):
    """
    Returns the default graph for testing, with fixed topology at this time
    """
    # nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1, 'central_wl': 1.55e-6}),
    #          1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
    #          2: WaveShaper(),
    #          3: DelayLine(),
    #          4: MeasurementDevice()
    #          }

    # edges = [(0,1, CorningFiber(parameters=[0])),
    #          (1,2, CorningFiber(parameters=[0])),
    #          (2,3),
    #          (3,4)]

    # nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1, 'central_wl': 1.55e-6}),
    #          1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
    #          2: WaveShaper(),
    #          3: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
    #          4: DelayLine(),
    #          5: WaveShaper(),
    #          6: MeasurementDevice()
    #          }

    # edges = [(0,1),
    #          (1,2),
    #          (2,3),
    #          (3,4),
    #          (4,5),
    #          (5,6)]

    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             1: WaveShaper(),
             2: WaveShaper(),
             3: MeasurementDevice()
            }
    edges = [(0, 1),
             (1, 2),
             (2, 3)]

    graph = Graph(nodes, edges, propagate_on_edges=True, deep_copy=deep_copy)
    graph.assert_number_of_edges()
    return graph


def get_propagator():
    return Propagator(window_t = 1e-8, n_samples = 2**14, central_wl=1.55e-6)


def get_evaluator(propagator):
    return RadioFrequencyWaveformGeneration(propagator)


def param_pool():
    graph = get_graph('limited')
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


def summation_no_copy(x):
    # return _summation_no_copy_recursive(x, 3)
    sum = 0

    for i in range(1, 3):
        sum += i * np.sum(x)
    return sum


def summation_copy(x):
    # return _summation_copy_recursive(x, 3)
    sum = 0

    for i in range(1, 3):
        sum += i * np.sum(copy.deepcopy(x))

    return sum

def _summation_no_copy_recursive(x, i):
    if i == 0:
        return 0
    return i * np.sum(x) + _summation_no_copy_recursive(x, i - 1)

def _summation_copy_recursive(x, i):
    if i == 0:
        return 0
    return i * copy.deepcopy(np.sum(x)) + _summation_copy_recursive(x, i - 1)


class SummationEvaluator(Evaluator):
    def evaluate_graph(self, graph, propagator):
        graph.propagate(propagator)
        state = graph.measure_propagator(len(graph.nodes) - 1)
        return np.sum(power_(state))


@pytest.fixture(scope='function')
def graph_deepcopy():
    return get_graph('full')


@pytest.fixture(scope='function')
def graph_no_deepcopy():
    return get_graph('none')


@pytest.fixture(scope='function')
def graph_limited_deepcopy():
    return get_graph('limited')


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


@pytest.mark.skip
@pytest.mark.parametrize("params", param_list)
def test_evaluation_equality(params, graph_deepcopy, graph_limited_deepcopy, graph_no_deepcopy, propagator, evaluator, nodeEdgeIndex_parameterIndex):
    assert get_individual_score(graph_limited_deepcopy, propagator, evaluator,
                                params, nodeEdgeIndex_parameterIndex[0],
                                nodeEdgeIndex_parameterIndex[1]) == \
           get_individual_score(graph_deepcopy, propagator, evaluator,
                                params, nodeEdgeIndex_parameterIndex[0],
                                nodeEdgeIndex_parameterIndex[1])
    
    assert get_individual_score(graph_limited_deepcopy, propagator, evaluator,
                                params, nodeEdgeIndex_parameterIndex[0],
                                nodeEdgeIndex_parameterIndex[1]) == \
           get_individual_score(graph_no_deepcopy, propagator, evaluator,
                                params, nodeEdgeIndex_parameterIndex[0],
                                nodeEdgeIndex_parameterIndex[1])

# @pytest.mark.skip
@pytest.mark.parametrize("params", param_list)
def test_gradient_equality(params, graph_deepcopy, graph_limited_deepcopy, graph_no_deepcopy, propagator, evaluator, nodeEdgeIndex_parameterIndex):
    # setup the gradient function and bounds
    fitness_funct_deepcopy = function_wrapper(graph_deepcopy, propagator, evaluator, exclude_locked=True)
    adam_fitness_funct_deepcopy = adam_function_wrapper(fitness_funct_deepcopy)
    fitness_grad_deepcopy = grad(adam_fitness_funct_deepcopy)
    
    propagator_no_deepcopy= get_propagator()
    fitness_funct_no_deepcopy = function_wrapper(graph_no_deepcopy, propagator_no_deepcopy, evaluator, exclude_locked=True)
    adam_fitness_funct_no_deepcopy = adam_function_wrapper(fitness_funct_no_deepcopy)
    fitness_grad_no_deepcopy = grad(adam_fitness_funct_no_deepcopy)

    propagator_limited_deepcopy= get_propagator()
    fitness_funct_limited_deepcopy = function_wrapper(graph_limited_deepcopy, propagator_limited_deepcopy, evaluator, exclude_locked=True)
    adam_fitness_funct_limited_deepcopy = adam_function_wrapper(fitness_funct_limited_deepcopy)
    fitness_grad_limited_deepcopy = grad(adam_fitness_funct_limited_deepcopy)

    print(fitness_grad_deepcopy(params, 0))
    print(fitness_grad_no_deepcopy(params, 0))
    print(fitness_grad_limited_deepcopy(params, 0))
    print(f'params: {params}')
    lowerbound, upperbound = graph_deepcopy.get_parameter_bounds()
    print(f'params lowerbound: {lowerbound}')
    print(f'params upperbound: {upperbound}')

    assert not np.allclose(fitness_grad_limited_deepcopy(params, 0), fitness_grad_deepcopy(params, 0))
    testing.assert_equal(fitness_grad_limited_deepcopy(params, 0), fitness_grad_no_deepcopy(params, 0))


@pytest.mark.skip
@pytest.mark.parametrize("params", param_list)
def test_gradient_summation_funct(params, graph_deepcopy, graph_limited_deepcopy, graph_no_deepcopy, propagator, nodeEdgeIndex_parameterIndex):
    evaluator = SummationEvaluator()

    # setup the gradient function and bounds
    fitness_funct_deepcopy = function_wrapper(graph_deepcopy, propagator, evaluator, exclude_locked=True)
    fitness_grad_deepcopy = grad(fitness_funct_deepcopy) # grad(adam_fitness_funct_deepcopy)
    
    propagator_no_deepcopy= get_propagator()
    fitness_funct_no_deepcopy = function_wrapper(graph_no_deepcopy, propagator_no_deepcopy, evaluator, exclude_locked=True)
    fitness_grad_no_deepcopy = grad(fitness_funct_no_deepcopy) # grad(adam_fitness_funct_no_deepcopy)

    propagator_limited_deepcopy = get_propagator()
    fitness_funct_limited_deepcopy = function_wrapper(graph_limited_deepcopy, propagator_limited_deepcopy, evaluator, exclude_locked=True)
    fitness_grad_limited_deepcopy = grad(fitness_funct_limited_deepcopy)

    print(fitness_grad_deepcopy(params))
    print(fitness_grad_no_deepcopy(params))
    print(fitness_grad_limited_deepcopy(params))
    print('param names:')
    print(graph_deepcopy.get_parameter_info())

    assert not np.allclose(fitness_grad_limited_deepcopy(params), fitness_grad_deepcopy(params))
    assert np.allclose(fitness_grad_limited_deepcopy(params), fitness_grad_no_deepcopy(params))


@pytest.mark.skip
@pytest.mark.parametrize("params", param_list)
def test_hessian_summation_funct(params, graph_deepcopy, graph_limited_deepcopy, graph_no_deepcopy, propagator, nodeEdgeIndex_parameterIndex):
    evaluator = SummationEvaluator()

    # setup the gradient function and bounds
    fitness_funct_deepcopy = function_wrapper(graph_deepcopy, propagator, evaluator, exclude_locked=True)
    fitness_hess_deepcopy = hessian(fitness_funct_deepcopy)
    
    propagator_no_deepcopy = get_propagator()
    fitness_funct_no_deepcopy = function_wrapper(graph_no_deepcopy, propagator_no_deepcopy, evaluator, exclude_locked=True)
    fitness_hess_no_deepcopy = hessian(fitness_funct_no_deepcopy)

    propagator_limited_deepcopy = get_propagator()
    fitness_funct_limited_deepcopy = function_wrapper(graph_limited_deepcopy, propagator_limited_deepcopy, evaluator, exclude_locked=True)
    fitness_hess_limited_deepcopy = hessian(fitness_funct_limited_deepcopy)

    params = np.array(params)
    print(fitness_hess_deepcopy(params))
    print(fitness_hess_no_deepcopy(params))
    print(fitness_hess_limited_deepcopy(params))
    print('param names:')
    print(graph_deepcopy.get_parameter_info())

    assert not np.allclose(fitness_hess_limited_deepcopy(params), fitness_hess_deepcopy(params))
    assert np.allclose(fitness_hess_limited_deepcopy(params), fitness_hess_no_deepcopy(params))


@pytest.mark.skip
@pytest.mark.parametrize("params", param_list)
def test_propagation_result_equality(params, graph_deepcopy, graph_limited_deepcopy, graph_no_deepcopy, propagator, evaluator, nodeEdgeIndex_parameterIndex):
    propagator_nodeepcopy = get_propagator()

    # deepcopy
    graph_deepcopy.distribute_parameters_from_list(params, nodeEdgeIndex_parameterIndex[0], nodeEdgeIndex_parameterIndex[1])
    graph_deepcopy.propagate(propagator)
    deepcopy_state = graph_deepcopy.measure_propagator(len(graph_deepcopy.nodes) - 1) # not as good a check but will do

    # not deepcopy
    graph_no_deepcopy.distribute_parameters_from_list(params, nodeEdgeIndex_parameterIndex[0], nodeEdgeIndex_parameterIndex[1])
    graph_no_deepcopy.propagate(propagator_nodeepcopy)
    no_deepcopy_state = graph_no_deepcopy.measure_propagator(len(graph_no_deepcopy.nodes) - 1)

    # limited deepcopy
    propagator_limited_deepcopy = get_propagator()
    graph_limited_deepcopy.distribute_parameters_from_list(params, nodeEdgeIndex_parameterIndex[0], nodeEdgeIndex_parameterIndex[1])
    graph_limited_deepcopy.propagate(propagator_limited_deepcopy)
    limited_deepcopy_state = graph_limited_deepcopy.measure_propagator(len(graph_limited_deepcopy.nodes) - 1)
    
    if GRAPHICAL_TESTING:
        display_states_deepcopy_noDeepcopy(deepcopy_state, no_deepcopy_state, propagator)

    testing.assert_allclose(limited_deepcopy_state, deepcopy_state)
    testing.assert_allclose(limited_deepcopy_state, no_deepcopy_state)


@pytest.mark.xfail
def test_autograd_deepcopy_behaviour():
    x = np.array([1, 2, 3, 4, 3.5, 920, 0])

    grad_copy = grad(summation_copy)
    grad_no_copy = grad(summation_no_copy)

    print(f'with deepcopy: {grad_copy(x)}')
    print(f'without deepcopy: {grad_no_copy(x)}')

    assert np.allclose(grad_copy(x), grad_no_copy(x))
