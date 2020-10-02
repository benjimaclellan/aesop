import autograd.numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.additive_noise import AdditiveNoise

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import EDFA, WaveShaper, PhaseModulator

from problems.example.assets.functions import power_, ifft_shift_, fft_, psd_

# TODO: figure out why noise levels seem too low? WATCH OUT FOR NORMALIZATION!!

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


@pytest.mark.skip
def test_static_vs_obj_runtime_filter():
    assert False 
    # TODO: do


# -------------------------------- Testing Photodiode -----------------------------------------

def get_laser_photodiode_graph(**kwargs):
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1e-4, 'central_wl': 1.55e-6, 'osnr_dB':20}),
             1: Photodiode(parameters_from_name=kwargs)
            }

    edges = [(0, 1)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


def get_laser_photodiode_graph_2(**kwargs):
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1e-4, 'central_wl': 1.55e-6, 'osnr_dB':20}),
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: WaveShaper(),
             3: EDFA(),
             4: Photodiode(parameters_from_name=kwargs)
            }

    edges = [(0, 1),
             (1, 2),
             (2, 3), 
             (3, 4)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


@pytest.fixture(scope='function')
def photodiode():
    return Photodiode()


@pytest.fixture(scope='function')
def photodiode_graph():
    return get_laser_photodiode_graph()


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_photodiode_basic(propagator, photodiode_graph):
    # tests the original graph with laser noise enabled
    photodiode_graph.propagate(propagator)
    photodiode_graph.inspect_state(propagator, freq_log_scale=True)
    photodiode_graph.display_noise_contributions(propagator, node=0) # after laser, unfiltered by photodiode
    photodiode_graph.display_noise_contributions(propagator, node=1) # filtered by photodiode, but also with photo noise once that's there
    assert False


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_photodiode_basic2(propagator):
    # tests the original graph with laser noise enabled
    photodiode_graph2 = get_laser_photodiode_graph_2()
    photodiode_graph2.propagate(propagator)

    amplifier_arr = photodiode_graph2.measure_propagator(3)
    photodiode_arr = photodiode_graph2.measure_propagator(4)
    _, ax = plt.subplots()
    ax.plot(propagator.t, (photodiode_arr - np.min(photodiode_arr)) / (np.max(photodiode_arr) - np.min(photodiode_arr)), label='Photodiode output voltage', lw=1)
    ax.plot(propagator.t, (amplifier_arr - np.min(amplifier_arr)) / (np.max(amplifier_arr) - np.min(amplifier_arr)), label='Pre-photodiode power', lw=1)
    ax.legend()
    plt.title('Photodiode effects')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Power and voltage, normalized between 0 and 1 (W and V)')
    plt.show()
    photodiode_graph2.inspect_state(propagator, freq_log_scale=True)
    photodiode_graph2.display_noise_contributions(propagator, node=4) # filtered by photodiode, but also with photo noise once that's there
    assert False


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_photodiode_no_input_noise(propagator, photodiode_graph):
    np.random.seed(1)
    # tests the original graph with laser noise enabled
    photodiode_graph.nodes[0]['model'].noise_model.noise_on = False
    photodiode_graph.propagate(propagator)
    photodiode_graph.inspect_state(propagator, freq_log_scale=True)
    photodiode_graph.display_noise_contributions(propagator, node=0) # after laser, unfiltered by photodiode
    photodiode_graph.display_noise_contributions(propagator, node=1) # filtered by photodiode, but also with photo noise once that's there
    assert False


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_photodiode_saturation(propagator, photodiode_graph):
    # TODO: confirm noise behaviour is expected
    _, ax = plt.subplots()
    # RMS powers
    P_total = []
    P_signal = []
    P_noise = []
    P_in = []
    # expected voltage cutoff when P_in = 6e-3 (default: I_d = R_d * P_in, with R_d = 0.5, I_d_max = 3e-3)
    # this should be at voltage = 3e-3 A * 50 Ohms = 0.15 V 
    for i in range(0, 100):
        P_in.append(i * 1e-4)
        photodiode_graph.nodes[0]['model'].parameters[0] = i*1e-4 # set different max powers for laser
        P_total.append(np.sqrt(np.mean(power_(photodiode_graph.get_output_signal(propagator)))))
        P_signal.append(np.sqrt(np.mean(power_(photodiode_graph.get_output_signal_pure(propagator)))))
        P_noise.append(np.sqrt(np.mean(power_(photodiode_graph.get_output_noise(propagator)))))
    
    ax.plot(np.array(P_in), np.array(P_total), label='total RMS voltage')
    ax.plot(np.array(P_in), np.array(P_signal), label='pure signal RMS voltage')
    ax.plot(np.array(P_in), np.array(P_noise), label='noise RMS voltage')
    ax.legend()
    ax.set_xlabel('Power in (W)')
    ax.set_ylabel('RMS voltage (V)')
    plt.title('Photodetector RMS voltage v. power in')
    plt.show()


# ---------------- random test ----------------------
@pytest.mark.skip
def test_lorentzian():
    f = np.linspace(-100, 100, num=2000)
    # HWHM = 1
    _, ax = plt.subplots(2, 1)
    for HWHM in [0.01, 0.5, 5, 10]:
        state_rf = 1 / np.pi * HWHM / ((f**2) + HWHM**2)
        state = np.fft.ifft(np.fft.ifftshift(state_rf))

        ax[0].plot(f, state, label=f'{HWHM}')
        ax[1].plot(f, state_rf, label=f'{HWHM}')
    
    ax[0].legend()
    ax[1].legend()
    plt.title('Lorentzian test')
    plt.show()


@pytest.mark.skip
def test_parseval(propagator):
    # https://www.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft
    # basically, our choice for fft_ and ifft_ normalization makes the INTEGRAL under the two curves equal
    signal = np.cos(2 * np.pi * 50e9 * propagator.t)
    energy_signal = np.sum(power_(signal)) * propagator.dt
    print(f'signal energy: {energy_signal}')
    signal_rf = fft_(signal, propagator.dt)
    energy_signal_rf = np.sum(power_(signal_rf)) * propagator.df
    print(f'signal_rf energy: {energy_signal_rf}')

    np.isclose(energy_signal, energy_signal_rf)


@pytest.mark.skip
def test_avg_normal(propagator):
    normal_dist = np.random.normal(scale=1, size=(propagator.n_samples, 2)).view(dtype='complex')
    print(np.mean(power_(normal_dist)))
    assert np.isclose(np.mean(power_(normal_dist / np.sqrt(2))), 1)

# -------------------------------- Testing Laser linewidth -----------------------------------------


def get_standard_graph(linewidth, coupling_eff=1):
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1e-4, 'central_wl': 1.55e-6, 'osnr_dB':20, 'FWHM_linewidth':linewidth}),
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: WaveShaper(),
             3: MeasurementDevice()
            }

    edges = [(0, 1),
             (1, 2),
             (2, 3)]

    graph = Graph(nodes, edges, propagate_on_edges=True, coupling_efficiency=coupling_eff)
    graph.assert_number_of_edges()
    return graph



# ------------------------------------ Test coupling efficiency ------------------------------
def phase_modulator_graph(coupling_eff):
    nodes = {0: ContinuousWaveLaser(peak_power=1, osnr_dB=2000, linewidth=1), # minimal noise in here, so the system will be mostly noise free
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             3: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             4: MeasurementDevice()
            }

    edges = [(0, 1),
             (1, 2),
             (2, 3),
             (3, 4)]

    graph = Graph(nodes, edges, propagate_on_edges=True, coupling_efficiency=coupling_eff)
    graph.assert_number_of_edges()
    return graph


def test_coupling_efficiency(propagator):
    for e in [0.9, 0.7, 0.5]:
        graph = phase_modulator_graph(e)
        graph.propagate(propagator)
        if not SKIP_GRAPHICAL_TEST:
            graph.inspect_state(propagator)
        
        output_power = power_(graph.measure_propagator(graph.get_output_node()))
        assert np.isclose(np.mean(output_power), e**(len(graph.nodes) - 1), atol=1e-3), f'mean power: {np.mean(output_power)}, expected power: { e**(len(graph.nodes) - 1)}'


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_coupling_efficiency_std_graph(propagator):
    for e in [0.96, 0.9, 0.85]:
        graph = get_standard_graph(1, coupling_eff=e)
        graph.propagate(propagator)
        graph.inspect_state(propagator)
