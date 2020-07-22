import autograd.numpy as np
import pytest
import random
import matplotlib.pyplot as plt

from problems.example.graph import Graph
from problems.example.assets.additive_noise import AdditiveNoise
from problems.example.assets.propagator import Propagator

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

SKIP_GRAPHICAL_TEST = True

@pytest.fixture(scope='function')
def propagator():
    return Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)


@pytest.fixture(scope='function')
def laser_graph():
    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             1: MeasurementDevice()
            }
    edges = [(0, 1)]
    graph = Graph(nodes, edges, propagate_on_edges=False)
    graph.assert_number_of_edges()
    return graph


@pytest.fixture(scope='function')
def default_graph():
    """
    Returns the default graph for testing, with fixed topology at this time
    """
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1, 'central_wl': 1.55e-6}),
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: WaveShaper(),
             3: MeasurementDevice()
            }

    edges = [(0, 1),
             (1, 2),
             (2, 3)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_laser_graph(laser_graph, propagator):
    laser_graph.propagate(propagator)
    laser_graph.inspect_state(propagator, freq_log_scale=True)

@pytest.mark.skip
def test_laser_osnr(laser_graph, propagator):
    signal = laser_graph.get_output_signal_pure(propagator)
    noise = laser_graph.get_output_noise(propagator)
    print(f'OSNR: {AdditiveNoise.get_OSNR(signal, noise)}')
    assert False


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_default_graph(default_graph, propagator):
    AdditiveNoise.simulate_with_noise = True
    default_graph.propagate(propagator, save_transforms=True)
    default_graph.inspect_state(propagator, freq_log_scale=True)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_default_graph_isolate_noise(default_graph, propagator):
    default_graph.display_noise_contributions(propagator)


# def test_plot():
#     param = np.array([1, 2, 3, 4, 5, 10, 20, 50, 100])
#     OSNR = np.array([-1.6017, -0.60744, 0.38948, 1.3936, 2.3968, 7.3985, 17.3932, 47.3924, 97.3937])
#     fig, ax = plt.subplots()
#     ax.plot(param, OSNR)
#     ax.scatter(param, OSNR)
#     plt.show()
