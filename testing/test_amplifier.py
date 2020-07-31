import autograd.numpy as np
import pytest

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import EDFA, WaveShaper


SKIP_GRAPHICAL_TEST = False


@pytest.fixture(scope='function')
def propagator():
    return Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)


@pytest.fixture(scope='function')
def edfa():
    return EDFA()


@pytest.fixture(scope='function')
def amp_graph():
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1e-4, 'central_wl': 1.55e-6, 'osnr_dB':20}),
             1: EDFA(),
             2: MeasurementDevice()
            }

    edges = [(0, 1),
             (1, 2)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_small_signal(propagator, edfa):
    edfa.display_small_signal_gain(propagator)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_gain(propagator, edfa):
    states = []
    for i in range(-9, 2):
        states.append(np.ones(propagator.n_samples).reshape(propagator.n_samples, 1) * 10**(i/2))
    edfa.display_gain(states, propagator)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_propagate(propagator, amp_graph):
    amp_graph.propagate(propagator)
    amp_graph.inspect_state(propagator)