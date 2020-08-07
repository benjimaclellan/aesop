import autograd.numpy as np
import pytest

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import EDFA, WaveShaper


SKIP_GRAPHICAL_TEST = False

def get_amp_graph(**kwargs):
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1e-4, 'central_wl': 1.55e-6, 'osnr_dB':20}),
             1: EDFA(parameters_from_name=kwargs),
             2: MeasurementDevice()
            }

    edges = [(0, 1),
             (1, 2)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph

@pytest.fixture(scope='function')
def propagator():
    return Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)


@pytest.fixture(scope='function')
def edfa():
    return EDFA()


@pytest.fixture(scope='function')
def amp_graph():
    return get_amp_graph()


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
    assert False


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_transform_visualisation(propagator, amp_graph):
    amp_graph.propagate(propagator, save_transforms=True)
    amp_graph.visualize_transforms([0, 1, 2], propagator)


@pytest.mark.skip
def test_gain_flatness():
    assert False 
    # TODO: verify that the gain flatness is indeed what I expect


@pytest.mark.skip
def test_static_vs_obj_runtime_filter():
    assert False 
    # TODO: do


def test_max_small_signal():
    pass 


def test_peak_wl():
    pass

# ----------------------- Test EDFA noise (also testing additive_noise class) -------------------

@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_noise_basic0(propagator):
    # tests the original graph by disabling laser noise (such that we can see the ASE noise by itself)
    graph = get_amp_graph()
    graph.nodes[0]['model'].noise_model.noise_on = False # bit of a hack but it's useful for development
    graph.propagate(propagator)
    graph.inspect_state(propagator)
    graph.display_noise_contributions(propagator)

    assert False


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_noise_basic1():
    # tests the original graph with laser noise  enabled
    pass

def test_increasing_small_gain():
    # test with increasing gain values, and confirm that ASE increases with gain
    pass

def test_increasing_noise_figures():
    # test with increasing noise figures, and confirm that ASE increases with noise figure
    pass

def test_increasing_power_max():
    # test with increasing max power, and confirm that ASE increases with noise figure
    pass

def test_noise_figure_consistent():
    # test that the noise figure just about lines up with expected (by doing the reverse calculation)
    # 1. Get ASE power
    # 2. Check that F(v) = P_ase / (hvG) + 1 / G, with P_ase being spectral density of ASE is approx valid (plot both)
    pass
