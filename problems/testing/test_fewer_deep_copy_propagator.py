import pytest
import autograd.numpy as np

from algorithms.parameter_optimization_utils import get_individual_score, get_initial_population

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator

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


@pytest.fixture(scope='module')
def graph_deepcopy():
    return get_graph(True)


@pytest.fixture(scope='module')
def graph_no_deepcopy():
    return get_graph(False)


@pytest.fixture(scope='module')
def propagator():
    return get_propagator()


@pytest.fixture(scope='module')
def evaluator(propagator):
    return get_evaluator(propagator)


@pytest.fixture(scope='module')
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
