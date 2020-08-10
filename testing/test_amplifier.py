import autograd.numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import EDFA, WaveShaper

from problems.example.assets.functions import power_, ifft_shift_


SKIP_GRAPHICAL_TEST = True

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
    for i in range(-9, -1):
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


# @pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_gain_flatness(edfa, propagator):
    # max diff in dB should be 1.5 dB, i.e. from 1000 to x = 10^(-1.5/10) * 1000 = 707.9
    # corresponds to EDFA default params
    peak_freq = speed_of_light / 1550e-9
    freq_low = speed_of_light / 1565e-9
    freq_high = speed_of_light / 1520e-9

    small_signal = ifft_shift_(edfa._get_small_signal_gain(propagator))
    line0_x = np.array([freq_low - propagator.central_frequency, freq_low - propagator.central_frequency])
    line1_x = np.array([freq_high - propagator.central_frequency, freq_high - propagator.central_frequency])
    line2_x = np.array([peak_freq - propagator.central_frequency, peak_freq - propagator.central_frequency])
    line_y = [0, 1000]

    _, ax = plt.subplots()
    ax.plot(propagator.f, small_signal)
    ax.plot(line0_x, line_y)
    ax.plot(line1_x, line_y)
    ax.plot(line2_x, line_y)
    plt.title('Small signal gain, with lines at the band edges')
    plt.show()


@pytest.mark.skip
def test_static_vs_obj_runtime_filter():
    assert False 
    # TODO: do


# ----------------------- Test EDFA noise (also testing additive_noise class) -------------------

@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_noise_basic0(propagator):
    # tests the original graph by disabling laser noise (such that we can see the ASE noise by itself)
    graph = get_amp_graph()
    graph.nodes[0]['model'].noise_model.noise_on = False # bit of a hack but it's useful for development
    graph.propagate(propagator)
    graph.inspect_state(propagator)
    graph.display_noise_contributions(propagator)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_noise_basic1(propagator):
    # tests the original graph with laser noise  enabled
    graph = get_amp_graph()
    graph.propagate(propagator)
    graph.inspect_state(propagator)
    graph.display_noise_contributions(propagator)

@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_increasing_noise_small_gain(propagator):
    # test with increasing gain values, and confirm that noise increases with gain
    # small signal gain increases, gain increases, noise increases
    _, ax = plt.subplots()
    for i in range(30, 20, -1):
        param = {'max_small_signal_gain_dB': i}
        graph = get_amp_graph(**param)
        graph.propagate(propagator)
        output = graph.measure_propagator(2)
        ax.plot(propagator.t, power_(output), label=f'small signal gain {i} dB')
    ax.legend()
    plt.title('Output, by small signal gain')
    plt.show()


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_increasing_ASE_small_gain(propagator):
    # test with increasing gain values, and confirm that noise increases with gain
    # small signal gain increases, gain increases, noise increases
    _, ax = plt.subplots()
    for i in range(30, 20, -3):
        param = {'max_small_signal_gain_dB': i}
        graph = get_amp_graph(**param)
        graph.nodes[0]['model'].noise_model.noise_on = False # disable laser noise, so ASE is only noise source
        output = graph.get_output_noise(propagator)
        ax.plot(propagator.t, power_(output), label=f'small signal gain {i} dB')
    ax.legend()
    plt.title('ASE noise, by small signal gain')
    plt.show()


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_increasing_noise_figures(propagator):
    # test with increasing noise figures, and confirm that ASE increases with noise figure
    _, ax = plt.subplots()
    for i in range(9, 2, -2):
        param = {'max_noise_fig_dB': i}
        graph = get_amp_graph(**param)
        graph.nodes[0]['model'].noise_model.noise_on = False # disable laser noise, so ASE is only noise source
        output = graph.get_output_noise(propagator)
        ax.plot(propagator.t, power_(output), label=f'Noise figure {i} dB')
    ax.legend()
    plt.title('ASE noise, by noise figure')
    plt.show()


# @pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_increasing_power_max(propagator):
    # test with increasing max power, and confirm that ASE increases with noise figure
    # larger P_max => larger gain => greater ASE
    _, ax = plt.subplots()
    for i in [10, 1, 0.1]:
        print(i)
        param = {'P_out_max': i}
        graph = get_amp_graph(**param)
        graph.nodes[0]['model'].noise_model.noise_on = False # disable laser noise, so ASE is only noise source
        output = graph.get_output_noise(propagator)
        ax.plot(propagator.t, power_(output), label=f'P_out_max {i} W')
    ax.legend()
    plt.title('ASE noise, by max output power')
    plt.show()


def test_noise_figure_consistent(edfa, propagator):
    # test that the noise figure just about lines up with expected (by doing the reverse calculation)
    # 1. Get ASE power
    # 2. Check that F(v) = P_ase / (hvG) + 1 / G, with P_ase being spectral density of ASE is approx valid (plot both)
    pass
