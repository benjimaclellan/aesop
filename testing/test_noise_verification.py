import autograd.numpy as np
import pytest
from autograd import grad
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light

from problems.example.assets.additive_noise import AdditiveNoise
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import power_, ifft_, ifft_shift_, fft_, psd_

from problems.example.graph import Graph
from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import EDFA, WaveShaper, PhaseModulator

"""
This document is intended to unite verification of noise into one document

Each test should (ideally) be run without a graphical component
"""
SKIP_GRAPHICAL_TEST = True


# ----------------- General Fixtures ---------------------------

@pytest.fixture(scope='function')
def propagator():
    return Propagator(window_t=1e-9, n_samples=2**14, central_wl=1.55e-6)


@pytest.fixture(autouse=True)
def setup():
    AdditiveNoise.simulate_with_noise = True


# ----------------- Basic AdditiveNoise class tests ---------------------

@pytest.fixture(scope='function')
def signal(propagator):
    return np.sin(2 * np.pi / 1e-9 * propagator.t) + 1j*0 # we just really want a visible signal that's not constant power so *shrugs*


@pytest.mark.AdditiveNoise
def test_incorrect_distribution():
    with pytest.raises(ValueError):
        AdditiveNoise(distribution='lorentzian')


@pytest.mark.AdditiveNoise
def test_incorrect_noise_type():
    with pytest.raises(ValueError):
        AdditiveNoise(noise_type='transcendent')


@pytest.mark.AdditiveNoise
def test_osnr(signal, propagator):
    for i in [1, 5, 10, 32]:
        noise = AdditiveNoise(noise_type='osnr', noise_param=i, seed=0)

        if (not SKIP_GRAPHICAL_TEST):
            noise.display_noisy_signal(signal, propagator)
            noise.display_noise(signal, propagator)

        noise_total = noise.add_noise_to_propagation(signal, propagator) - signal
        assert np.isclose(AdditiveNoise.get_OSNR(signal, noise_total), i)


@pytest.mark.AdditiveNoise
def test_absolute_noise_power(propagator, signal):
    noise = AdditiveNoise(noise_param=1, seed=0, noise_type='absolute power')
    
    zero_signal = np.zeros(propagator.n_samples).reshape((propagator.n_samples, 1))
    noisy_signal = noise.add_noise_to_propagation(zero_signal, propagator)
    assert np.isclose(np.mean(power_(noisy_signal)), 1)

    noise_total = noise.add_noise_to_propagation(signal, propagator) - signal
    assert np.isclose(np.mean(power_(noise_total)), 1)


@pytest.mark.AdditiveNoise
def test_multiple_noise_sources_add_before_use(signal, propagator):
    noise = AdditiveNoise(noise_type='absolute power', noise_param=16, seed=0)
    noise.add_noise_source(noise_type='absolute power', noise_param=8) 
    noise.add_noise_source(noise_type='absolute power', noise_param=4)
    noise.add_noise_source(noise_type='absolute power', noise_param=2)
    
    if not SKIP_GRAPHICAL_TEST:
        noise.display_noisy_signal(signal, propagator)
        noise.display_noise(signal, propagator)
        noise.display_noise_sources_absolute(propagator)

    noise_total = noise.add_noise_to_propagation(signal, propagator) - signal

    # there might be noise interference so we allow a larger tolerance
    assert np.isclose(np.mean(power_(noise_total)), 2 + 4 + 8 + 16, atol=1)


@pytest.mark.AdditiveNoise
def test_multiple_noise_sources_add_after_use(signal, propagator):
    noise = AdditiveNoise(noise_type='absolute power', noise_param=16, seed=0)
    
    if not SKIP_GRAPHICAL_TEST:
        noise.display_noisy_signal(signal, propagator=propagator)
        noise.display_noise_sources_absolute(propagator=propagator)

    noise_total = noise.add_noise_to_propagation(signal, propagator) - signal
    assert np.isclose(np.mean(power_(noise_total)), 16)

    noise.add_noise_source(noise_type='absolute power', noise_param=8) 
    noise.add_noise_source(noise_type='absolute power', noise_param=4)
    noise.add_noise_source(noise_type='absolute power', noise_param=2)
    
    if not SKIP_GRAPHICAL_TEST:
        noise.display_noisy_signal(signal, propagator=propagator)
        noise.display_noise_sources_absolute(propagator=propagator)

    noise_total = noise.add_noise_to_propagation(signal, propagator) - signal

    # there might be noise interference so we allow a larger tolerance
    assert np.isclose(np.mean(power_(noise_total)), 2 + 4 + 8 + 16, atol=1) 


@pytest.mark.AdditiveNoise
def test_noise_disactivation(signal, propagator):
    noise = AdditiveNoise(noise_param=4, seed=0)
    if not SKIP_GRAPHICAL_TEST:
        noise.display_noisy_signal(signal, propagator=propagator) # should show noisy signal
    noise_total = noise.add_noise_to_propagation(signal, propagator) - signal
    assert not np.isclose(np.mean(noise_total), 0)

    noise.noise_on = False
    if not SKIP_GRAPHICAL_TEST:
        noise.display_noisy_signal(signal, propagator=propagator) # should show noisy signal
    noise_total = noise.add_noise_to_propagation(signal, propagator) - signal
    assert np.allclose(noise_total, 0)
   
    noise.noise_on = True
    if not SKIP_GRAPHICAL_TEST:
        noise.display_noisy_signal(signal, propagator=propagator) # should show noisy signal
    noise_total = noise.add_noise_to_propagation(signal, propagator) - signal
    assert not np.isclose(np.mean(noise_total), 0)

    AdditiveNoise.simulate_with_noise = False
    if not SKIP_GRAPHICAL_TEST:
        noise.display_noisy_signal(signal, propagator=propagator) # should show pure signal
    noise_total = noise.add_noise_to_propagation(signal, propagator) - signal
    assert np.allclose(noise_total, 0)
  

@pytest.mark.AdditiveNoise
def test_incompatible_signal_shapes(signal, propagator):
    noise = AdditiveNoise()
    noise.add_noise_to_propagation(signal, propagator)
    with pytest.raises(ValueError):
        noise.add_noise_to_propagation(signal[0:100], propagator)


@pytest.mark.AdditiveNoise
def test_real_signal_generation_freq(propagator):
    real_signal = AdditiveNoise._get_real_noise_signal_freq(propagator)
    
    if not SKIP_GRAPHICAL_TEST:
        start = propagator.n_samples // 2 - 100
        stop = propagator.n_samples // 2 + 101

        _, ax = plt.subplots(2, 1)
        ax[0].plot(propagator.f[start:stop], power_(np.fft.fftshift(real_signal))[start:stop])
        ax[0].set_ylabel('Noise power')
        ax[0].set_xlabel('Frequency (Hz)')
        ax[1].plot(propagator.f[start:stop], np.angle(np.fft.fftshift(real_signal))[start:stop])
        ax[1].set_ylabel('Noise phase')
        ax[1].set_xlabel('Frequency (Hz)')

        plt.show()

    assert np.allclose(np.imag(np.fft.ifft(real_signal, axis=0)), 0, atol=1e-12)


@pytest.mark.AdditiveNoise
def test_seeding(signal, propagator):
    noise1 = AdditiveNoise(noise_type='absolute power', seed=101)
    noise2 = AdditiveNoise(noise_type='absolute power', seed=3030)
    
    output1 = noise1.add_noise_to_propagation(signal, propagator)
    output2 = noise2.add_noise_to_propagation(signal, propagator)

    assert not np.allclose(output1, output2)


@pytest.mark.AdditiveNoise
def test_resampling(signal, propagator):
    noise = AdditiveNoise(noise_type='absolute power', seed=0)
    
    output1 = noise.add_noise_to_propagation(signal, propagator)
    
    noise.resample_noise(seed=1)
    output2 = noise.add_noise_to_propagation(signal, propagator)

    assert not np.allclose(output1, output2)

# ----------------- CW Laser tests ---------------------

def get_laser_only_graph(peak_power=1e-3, osnr_dB=55, linewidth=1e3):
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': peak_power, 'central_wl': 1.55e-6, 'osnr_dB':osnr_dB, 'FWHM_linewidth':linewidth}),
             1: MeasurementDevice()
            }

    edges = [(0, 1)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph

@pytest.mark.CWL
def test_peak_power(propagator):
    """
    Check that peak power averages out correctly (with moderate noise)
    """
    for i in range(5):
        graph = get_laser_only_graph(peak_power=10**(-1 * i))
        graph.propagate(propagator)
        output = graph.measure_propagator(graph.get_output_node())
        assert np.isclose(np.mean(power_(output)), 10**(-1 * i), atol=1e-3)


@pytest.mark.CWL 
def test_zero_linewidth(propagator):
    """
    Check that zero linewidth and very large osnr gives us essentially an ideal graph
    (i.e. confirm that in the limit of noise disappearing, the behaviour is as expected)
    """
    graph = get_laser_only_graph(osnr_dB=1000, linewidth=0)
    graph.propagate(propagator)
    output1 = graph.measure_propagator(graph.get_output_node())
    if not SKIP_GRAPHICAL_TEST:
        graph.inspect_state(propagator, freq_log_scale=True)
    
    AdditiveNoise.simulate_with_noise = False
    graph.propagate(propagator)
    output2 = graph.measure_propagator(graph.get_output_node())

    if not SKIP_GRAPHICAL_TEST:
        graph.inspect_state(propagator, freq_log_scale=False)
    
    assert np.allclose(output1, output2), f'output1: {output1}, output2: {output2}'


@pytest.mark.CWL 
def test_osnr_with_zero_linewidth(propagator):
    for i in range(20, 55, 5):
        graph = get_laser_only_graph(peak_power=1e-3, osnr_dB=i, linewidth=0)
        if not SKIP_GRAPHICAL_TEST:
            graph.propagate(propagator)
            graph.inspect_state(propagator, freq_log_scale=True)

        osnr = AdditiveNoise.get_OSNR(graph.get_output_signal_pure(propagator),
                                      graph.get_output_noise(propagator))
        
        assert np.isclose(osnr, i)


@pytest.mark.CWL
def test_linewidth_with_osnr_large(propagator):
    # TODO: refactor this test to have a non-graphical test
    for i in [1e9, 5e9, 10e9, 50e9, 100e9]:
        graph = get_laser_only_graph(osnr_dB=1000, linewidth=i)
        if SKIP_GRAPHICAL_TEST:
            graph.propagate(propagator)
            output = graph.measure_propagator(1)

            _, ax = plt.subplots()
            psd = psd_(output, propagator.dt, propagator.df)
            h0 = i / np.pi
            lineshape = 1e-3 * h0 / (np.power(propagator.f, 2) + np.power(np.pi * h0 / 2, 2)) # dt is just normalization stuff
            ax.plot(propagator.f, psd / np.max(psd), label='psd')
            ax.plot(propagator.f, lineshape / np.max(lineshape), label='lineshape', ls=':')
            ax.legend()
            plt.show()

# ----------------- EDFA tests ---------------------
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


def get_amp_graph_2(**kwargs):
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1e-4, 'central_wl': 1.55e-6, 'osnr_dB':20}),
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: WaveShaper(),
             3: EDFA(parameters_from_name=kwargs),
             4: MeasurementDevice()
            }

    edges = [(0, 1),
             (1, 2),
             (2, 3), 
             (3, 4)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


def get_edfa(**params):
    graph = get_amp_graph(**params)
    return graph.nodes[1]['model']


@pytest.mark.EDFA
def test_small_signal_peak(propagator):
    for i in range(20, 31):
        param = {'max_small_signal_gain_dB':i}
        edfa = get_edfa(**param)
        small_signal = edfa._get_small_signal_gain(propagator)
        if not SKIP_GRAPHICAL_TEST and i % 10 == 0:
            _, ax = plt.subplots()
            ax.plot(propagator.f, ifft_shift_(small_signal))
            ax.set_xlabel('Frequency from carrier (Hz)')
            ax.set_ylabel('Small signal gain (pure ratio, not dB)')
            plt.title(f'Small signal gain {i} dB')
            plt.show()

        assert np.isclose(np.max(small_signal), 10**(i/10))


@pytest.mark.EDFA
def test_small_signal_flatness(propagator):
    for i in range(3, 17):
        param = {'gain_flatness_dB':i, 'band_lower':1520e-9, 'band_upper':1565e-9, 'peak_wl':1550e-9}
        centre_freq =  speed_of_light / (1550e-9)
        far_freq = speed_of_light / (1520e-9)
        freq_dist = far_freq - centre_freq
        target_index = np.round(freq_dist / propagator.df).astype(int)

        edfa = get_edfa(**param)
        small_signal = edfa._get_small_signal_gain(propagator)
        if not SKIP_GRAPHICAL_TEST and i % 8 == 0:
            _, ax = plt.subplots()
            ax.plot(propagator.f, ifft_shift_(small_signal))
            ax.plot(np.array([-freq_dist, -freq_dist]), np.array([0, np.max(small_signal)]))
            ax.set_xlabel('Frequency from carrier (Hz)')
            ax.set_ylabel('Small signal gain (pure ratio, not dB)')
            plt.title(f'Gain flatness {i} dB')
            plt.show()

        assert np.isclose(np.max(small_signal) / small_signal[target_index, 0], 10**(i/10), atol=1e-1)


@pytest.mark.EDFA
@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
@pytest.mark.parametrize('noise_use', [False, True])
def test_propagate(propagator, noise_use):
    amp_graph = get_amp_graph()
    AdditiveNoise.simulate_with_noise = noise_use
    amp_graph.propagate(propagator)
    amp_graph.inspect_state(propagator, freq_log_scale=True)
    amp_graph.propagate(propagator)
    amp_graph.inspect_state(propagator, freq_log_scale=True, title='Laser followed by amplifier output')


@pytest.mark.EDFA
@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
@pytest.mark.parametrize('noise_use', [False, True])
def test_propagate2(propagator, noise_use):
    AdditiveNoise.simulate_with_noise = noise_use
    amp_graph = get_amp_graph_2()
    amp_graph.propagate(propagator)
    amp_graph.inspect_state(propagator, freq_log_scale=True)
    amp_graph.display_noise_contributions(propagator, title='CWL -> phase modulator -> WaveShaper -> EDFA output')


@pytest.mark.EDFA
@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_transform_visualisation(propagator):
    amp_graph = get_amp_graph()
    amp_graph.propagate(propagator, save_transforms=True)
    amp_graph.visualize_transforms([0, 1, 2], propagator)


@pytest.mark.EDFA
@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_small_signal(propagator, edfa):
    edfa.display_small_signal_gain()


@pytest.mark.EDFA
@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_gain_curve(propagator, edfa):
    states = []
    for i in range(-9, -2):
        states.append(np.ones(propagator.n_samples).reshape(propagator.n_samples, 1) * 10**(i/2.1))
    edfa.display_gain(states, propagator)


@pytest.mark.EDFA
@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_gain_max(propagator, edfa):
    states = []

    power_in = np.array([1e-3 * 10**(i / 10) for i in np.arange(-40, 5, 0.1)])
    states = [np.ones(propagator.n_samples).reshape(propagator.n_samples, 1) * np.sqrt(P_in) for P_in in power_in]

    gain_level = np.array([np.max(edfa._gain(state, propagator)) for state in states])

    _, ax = plt.subplots()
    ax.plot(10 * np.log10(power_in / (1e-3)), 10 * np.log10(gain_level))
    ax.set_xlabel('Power in (dBm)')
    ax.set_ylabel('Gain (dB)')
    plt.title('Gain vs input power of EDFA')
    plt.show()

# ----------------- EDFA tests (preliminary Variability-Sensitivity Analysis) ---------------------
@pytest.mark.EDFA
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


@pytest.mark.EDFA
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


@pytest.mark.EDFA
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


@pytest.mark.EDFA
@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
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


@pytest.mark.EDFA
@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_dB_plots(propagator):
    edfa_params = {'max_noise_fig_dB': 3.5, 'max_small_signal_gain_dB': 40, 'P_out_max':1}
    graph = get_amp_graph(**edfa_params)
    ASE = graph.get_output_noise(propagator, save_transforms=True)
    gain = graph.nodes[1]['model'].transform[0][1] # nab the transform, which is gain

    _, ax = plt.subplots()
    ax.plot(propagator.f, 10 * np.log10(np.fft.fftshift(fft_(ASE, propagator.dt))), label='ASE')
    ax.plot(propagator.f, gain, label='gain')
    ax.legend()

    plt.title(f'ASE and gain, with max noise fig 3.5, max small signal gain 40')
    plt.show()

# ------------------- Modulator Check --------------------

def get_phase_modulator_graph(phase_noise_points=None, **kwargs):
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1e-4, 'central_wl': 1.55e-6, 'osnr_dB':20}),
             1: PhaseModulator(phase_noise_points=phase_noise_points, parameters_from_name=kwargs),
             2: MeasurementDevice()
            }

    edges = [(0, 1),
             (1, 2)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


@pytest.mark.skip
@pytest.mark.PhaseModulator
def test_modulator_behaviour(propagator):
    AdditiveNoise.simulate_with_noise = False 
    param = {'depth':2 * np.pi}
    graph = get_phase_modulator_graph(**param)
    graph.propagate(propagator)
    graph.inspect_state(propagator)

    AdditiveNoise.simulate_with_noise = True 


@pytest.mark.PhaseModulator
def test_phase_noise_interpolation(propagator):
    noise_test_points = [(10e3, -116), (100e3, -117), (1e6, -118), (10e6, -140)]
    graph = get_phase_modulator_graph(phase_noise_points=noise_test_points)
    graph.propagate(propagator)
    graph.inspect_state(propagator)


@pytest.mark.PhaseModulator
def test_add_linewidth_to_centre_freq(propagator):
    for FWHM in [1, 13e3, 1e9, 5e9, 10e9]:
        noise_model = AdditiveNoise(noise_type='phase noise from linewidth', noise_param=FWHM, noise_on=False)
        phase_noise = noise_model.get_phase_noise(propagator)

        _, ax = plt.subplots(3, 1)
        ax[0].plot(propagator.t, phase_noise)
        ax[0].set_xlabel('time (s)')
        ax[0].set_ylabel('Phase noise (rad)')

        frequency = 12e9
        transform = np.cos(2 * np.pi * frequency * propagator.t + phase_noise)
        ax[1].plot(propagator.t, transform)
        ax[1].set_xlabel('time (s)')
        ax[1].set_ylabel('Transform (no units, is the modulation)')

        psd = psd_(transform, propagator.dt, propagator.df)
        h0 = FWHM / np.pi
        lineshape = 1e-3 * h0 / (np.power(propagator.f, 2) + np.power(np.pi * h0 / 2, 2)) # dt is just normalization stuff
        ax[2].plot(propagator.f, psd / np.max(psd), label='psd')
        ax[2].plot(propagator.f, lineshape / np.max(lineshape), label='lineshape', ls=':')
        ax[2].legend()
        ax[2].set_xlabel('frequency offset (Hz)')
        ax[2].set_ylabel('Transform (i.e. the modulation)')
        plt.show()

# ----------------- EDFA tests ---------------------




    

# ------------------- out of commission laser test
@pytest.mark.skip
def test_linewidth(propagator):
    np.random.seed(100)

    N = 100

    for j in range(3):
        print(j)
        laser_graph = get_laser_only_graph(linewidth=2e9 * 10**j)
        average_state = np.zeros(propagator.n_samples, dtype='complex').reshape((propagator.n_samples, 1))

        for i in range(N):
            laser_graph.propagate(propagator)
            average_state += laser_graph.measure_propagator(node=1)
            if (i == 0):
                laser_graph.inspect_state(propagator)
            laser_graph.resample_all_noise(seed=i)

        average_state /= N
        _, ax = plt.subplots(2, 1)
        ax[0].plot(propagator.t, power_(average_state))
        ax[1].plot(propagator.f, psd_(average_state, propagator.dt, propagator.df))
        plt.title('Average linewidth results')
        plt.show()


@pytest.mark.skip
def test_linewidth_effects(propagator):
    np.random.seed(100)

    for j in range(3):
        laser_graph = get_standard_graph(2e9 * 10**j)
        laser_graph.propagate(propagator)
        laser_graph.inspect_state(propagator) # , freq_log_scale=True)
        osnr = AdditiveNoise.get_OSNR(laser_graph.get_output_signal_pure(propagator), laser_graph.get_output_noise(propagator))
        laser_graph.display_noise_contributions(propagator, node=0, title=f'osnr: {osnr}, linewidth: {2 * 10**j} GHz')


# ---------------------------------- AUTOGRAD TEST SLUSH PILE FOR NOW --------------------------------------
def gain_sum_no_noise(signal):
    return np.sum(np.abs(5 * signal))


def get_funct_gain_sum_noise(propagator):
    def gain_sum_noise(signal):
        gain = 5 * signal
        noise = AdditiveNoise(noise_param=4, seed=0)
        output = noise.add_noise_to_propagation(gain, propagator)
        return np.sum(np.abs(output))
    return gain_sum_noise


def get_funct_gain_separate_sum_noise(test_signal, propagator):
    noise = AdditiveNoise(noise_param=4, seed=0)
    noise.add_noise_to_propagation(test_signal, propagator) # try without this too...
                                                # But in practice if we evaluate before we derive
                                                # this test is accurate to what would happen
    def funct(signal):
        gain = 5 * signal
        output = noise.add_noise_to_propagation(gain, propagator)
        return np.sum(np.abs(output))
    
    return funct

@pytest.mark.skip
def test_autograd_gain1(signal, propagator):
    tmp = np.copy(signal)
    print(f'Sum power no noise: {gain_sum_no_noise(tmp)}')
    
    tmp = np.copy(signal)
    gain_sum_noise = get_funct_gain_sum_noise(propagator)
    print(f'Sum power noise: {gain_sum_noise(tmp)}')
    
    tmp = np.copy(signal)
    gain_separate_noise_gen = get_funct_gain_separate_sum_noise(tmp, propagator)
    tmp = np.copy(signal)
    print(f'Sum power separate noise: {gain_separate_noise_gen(tmp)}')

    grad_no_noise = grad(gain_sum_no_noise)
    grad_noise = grad(gain_sum_noise)
    grad_separate_noise_gen = grad(gain_separate_noise_gen)

    tmp = np.copy(signal)
    print(f'grad sum power no noise: {grad_no_noise(tmp)}')
    
    tmp = np.copy(signal)
    print(f'grad sum power noise: {grad_noise(tmp)}')

    tmp = np.copy(signal)
    print(f'grad sum power separate noise: {grad_separate_noise_gen(tmp)}')

    assert False


@pytest.mark.skip
def test_autograd_convergence(signal, propagator):
    # same as the gain test, but we'll take the average of a BUNCH of noisy signals and see if it converges
    TOTAL_ITERATIONS = 1

    grad_sum = np.zeros((signal.shape[0], 1), dtype='complex')
    noise = AdditiveNoise(noise_param=4, seed=0, noise_on=True, noise_type='absolute power')
    noise.add_noise_to_propagation(np.copy(signal), propagator)

    for i in range(TOTAL_ITERATIONS):
        tmp = np.copy(signal)
        noise.resample_noise(seed=i)
        
        def funct(x):
            """
            Making this a linear function bc it's easier to validate the derivative there
            """
            gain = 5 * x
            output = noise.add_noise_to_propagation(gain, propagator)
            return np.sum(output).astype(dtype='float')
        
        grad_funct = grad(funct)
        print("About to evaluate grad function!")
        grad_sum += grad_funct(tmp)

    avg_grad = grad_sum / TOTAL_ITERATIONS
    print("Average gradient:")
    print(avg_grad)
    assert False


@pytest.mark.skip
def test_autograd_gain_signal_absolute(propagator):
    signal = np.zeros((propagator.n_samples, 1), dtype='complex')
    noise = AdditiveNoise(noise_param=4, seed=0, noise_on=True, noise_type='absolute power')
    noise.add_noise_to_propagation(signal, propagator)

    noise.resample_noise(seed=0)
    def funct(G):
        all_noise = noise.add_noise_to_propagation(signal, propagator) # using empty signal but doesn't matter, it's a constant with this deriv
        print(f"all_noise sum: {np.sum(np.abs(all_noise))}")
        return np.sum(np.abs(G * all_noise))
    
    grad_funct = grad(funct)
    print("Evaluating grad function")
    gradient_at_G = grad_funct(np.array([5]))
    print(gradient_at_G)

    # get noise sum
    noise.resample_noise(seed=0)
    noise_sum = np.sum(np.abs(noise.add_noise_to_propagation(signal, propagator)))
    assert np.isclose(gradient_at_G[0], noise_sum)


@pytest.mark.skip
def test_autograd_gain_signal_relative(propagator):
    GAIN = 8

    signal = np.ones((propagator.n_samples, 1), dtype='complex')
    noise = AdditiveNoise(noise_param=4, seed=0, noise_on=True, noise_type='osnr')
    noise.add_noise_to_propagation(signal, propagator)

    noise.resample_noise(seed=0)
    def funct(G):
        output = noise.add_noise_to_propagation(G * signal, propagator) -  G * signal
        return np.sum(output).astype(dtype='float')
    
    grad_funct = grad(funct)
    print("Evaluating grad function")
    gradient_at_G = grad_funct(np.array([GAIN]))
    print(gradient_at_G)

    # test with expected val
    noise.resample_noise(seed=0)
    noise_sum = np.sum(noise.add_noise_to_propagation(signal, propagator) - signal).astype(dtype='float')
    print(f"noise_sum: {noise_sum}")
    assert np.isclose(gradient_at_G[0], noise_sum)


# ----------------- Miscellaneous tests ---------------------
