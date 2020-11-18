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
from problems.example.node_types_subclasses.single_path import OpticalAmplifier, WaveShaper, PhaseModulator

SKIP_GRAPHICAL_TEST = True

def dBc_to_power(dBc, reference_power):
    """
    Return power of a waveband in Watts from its dBc based on reference power reference_power

    :param dBc: decibels relative to the carrier
    :param reference power: carrier / reference signal power. This is usually total signal power
    """
    return 10**(dBc / 10) * reference_power


def W_to_dBm(power):
    return 10 * np.log10(power / (1e-3))


def dBm_to_W(dBm):
    return (1e-3) * 10**(dBm/10)

def plot_psd_v_wavelength(signal, propagator, title='', dB=True):
    amplitude = np.fft.fftshift(np.abs(fft_(signal, propagator.dt)))
    amplitude = amplitude / np.max(amplitude)
    wavelengths = speed_of_light / (propagator.f + propagator.central_frequency) * 1e9

    if dB:
        ylabel = 'Amplitude (dB)'
        amplitude = 10 * np.log10(amplitude)
    else:
        ylabel = 'Amplitude (a.u.)'
    _, ax = plt.subplots()
    ax.plot(wavelengths, amplitude)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_lineshape_dB(signal, propagator, freq_range, title=''):
    psd = W_to_dBm(power_(np.fft.fftshift(fft_(signal, propagator.dt))))

    _, ax = plt.subplots()
    ax.plot(propagator.f, psd)
    ax.set_xlim(freq_range[0], freq_range[1])
    ax.set_ylim(-150, 0.5 * np.max(psd))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dBm)')
    plt.title(title)
    plt.show()


def plot_lineshape(signal, propagator, freq_range, linewidth, title=''):
    psd = psd_(signal, propagator.dt, propagator.df)
    psd = psd / np.max(psd)
    h0 = linewidth / np.pi
    lineshape = 1e-3 * h0 / (np.power(propagator.f, 2) + np.power(np.pi * h0 / 2, 2))
    _, ax = plt.subplots()
    ax.plot(propagator.f, psd, label='Power distribution')
    ax.plot(propagator.f, lineshape / np.max(lineshape), label='Expected Lineshape', ls=':')
    ax.set_xlim(freq_range[0], freq_range[1])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u.)')
    ax.legend()
    plt.title(title)
    plt.show()


def plot_phase_psd_v_freq(signal0, signal1, propagator, freq_range, FWHM, title=''):
    # as a quick explanation: we can get phase noise in dBc/Hz via the methods usually experimentally used
    # Take a sideband (we'll take one frequency bucket), measure the power, normalize to 1 Hz
    # Then, we can convert dBc/Hz to rad^2/Hz -> rad/sqrt(Hz) using the power of AdditiveNoise.dBc_per_Hz_to_rad2_per_Hz
    # THEN we take the log and plot!
    total_power_0 = np.sum(power_(signal0)) # this is the carrier
    dBc_per_Hz_0 = 10 * np.log10(psd_(signal0, propagator.dt, propagator.df) / total_power_0)
    total_power_1 = np.sum(power_(signal1)) # this is the carrier
    dBc_per_Hz_1 = 10 * np.log10(psd_(signal1, propagator.dt, propagator.df) / total_power_1)
    rad_per_sqrt_Hz_0 = np.sqrt(AdditiveNoise.dBc_per_Hz_to_rad2_per_Hz(dBc_per_Hz_0))
    rad_per_sqrt_Hz_1 = np.sqrt(AdditiveNoise.dBc_per_Hz_to_rad2_per_Hz(dBc_per_Hz_1))
    dB_rad_per_sqrt_0 = 10 * np.log10(rad_per_sqrt_Hz_0)
    dB_rad_per_sqrt_1 = 10 * np.log10(rad_per_sqrt_Hz_1)

    h0 = FWHM / np.pi
    expected_phase_psd = 10 * np.log10(np.sqrt(h0 / 2 / np.power(propagator.f, 2)))

    _, ax = plt.subplots()
    ax.plot(np.log10(propagator.f), dB_rad_per_sqrt_0, label='ADJUSTIK E15, simulated')
    ax.plot(np.log10(propagator.f), dB_rad_per_sqrt_1, label='ADJUSTIK X15, simulated')
    ax.plot(np.log10(propagator.f), expected_phase_psd, label='expected phase psd (from Lorentzian linewidth)', ls=':')
    ax.set_xlim(1, np.log10(freq_range[1]))
    ax.set_xlabel('log10(Frequency) (log10(Hz))')
    # ax.set_ylim(np.min([np.min(rad_per_sqrt_Hz_0), np.min(rad_per_sqrt_Hz_1), np.min(expected_phase_psd)]), 
    #             np.max([np.max(rad_per_sqrt_Hz_0), np.max(rad_per_sqrt_Hz_1), np.max(expected_phase_psd)]))
    ax.legend()
    plt.title(title)
    plt.show()

# --------------------- Laser Validation ------------------
def laser_test_graph(**laser_params):
    # NOTE: we use the measurement device instead of a photodiode because the data we're looking at was made with a spectrum analyzer of high res
    nodes = {0: ContinuousWaveLaser(parameters_from_name=laser_params),
             1: MeasurementDevice()
            }

    edges = [(0, 1)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


def get_SFL1550_graph_propagator():
    osnr_db = 1000
    # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=4934
    # NOTE: OSNR was not provided and was guesstimated from the graph by yours truly (Julie)
    laser_params = {'peak_power':40e-3, 'central_wl':1550e-9, 'osnr_dB':osnr_db, 'FWHM_linewidth':50e3}

    # We want to see a wavelength range between 1548 nm to 1552 nm for comparison to the given graph.
    # That's equivalent to a frequency diff of: 
    f_mid = speed_of_light / 1550e-9
    f_max = speed_of_light / 1548e-9
    f_min = speed_of_light / 1552e-9
    f_one_sided_range = max(f_mid - f_min, f_max - f_mid)

    # now, we know f_one_sided_range = N / time_window / 2
    N = 2**24 # 2**24 # this was the resolution necessary to assess that linewidth seemed correct
    time_window = N / f_one_sided_range / 2
    propagator = Propagator(n_samples=N, window_t=time_window, central_wl=1550e-9)
    print(f'propagator: (n_samples, window_t, dt, df): {propagator.n_samples}, {propagator.window_t}, {propagator.dt}, {propagator.df}')
    return laser_test_graph(**laser_params), propagator


def get_ORS1500_graph_propagator():
    osnr_dB = 70 # a reasonable value, since no indication is given

    laser_params = {'peak_power':5e-3, 'central_wl':1542e-9, 'osnr_dB':osnr_dB, 'FWHM_linewidth':0.5}

    # we want to see a frequency range of -2 Hz to 2 Hz
    N = 2**14
    time_window = N / 2 / 2 # time range, as to get desired frequency range
    propagator = Propagator(n_samples=N, window_t=time_window, central_wl=1542e-9)
    print(f'propagator: (n_samples, window_t, dt, df): {propagator.n_samples}, {propagator.window_t}, {propagator.dt}, {propagator.df}')
    return laser_test_graph(**laser_params), propagator


def get_ADJUSTIK_E15_graph_propagator():
    osnr_dB = 55 # given
    laser_params = {'peak_power':40e-3, 'central_wl':1550e-9, 'osnr_dB':osnr_dB, 'FWHM_linewidth':0.1e3}

    # we want to see a frequency range up to 10e3 Hz
    N = 2**14
    time_window = N / (10e3) / 2
    propagator = Propagator(n_samples=N, window_t=time_window, central_wl=1550e-9)
    print(f'propagator: (n_samples, window_t, dt, df): {propagator.n_samples}, {propagator.window_t}, {propagator.dt}, {propagator.df}')
    return laser_test_graph(**laser_params), propagator


def get_ADJUSTIK_X15_graph_propagator():
    osnr_dB = 55 # given
    laser_params = {'peak_power':22.5e-3, 'central_wl':1550e-9, 'osnr_dB':osnr_dB, 'FWHM_linewidth':0.1e3}

    # we want to see a frequency range up to 10e3 Hz
    N = 2**14
    time_window = N / (10e3) / 2
    propagator = Propagator(n_samples=N, window_t=time_window, central_wl=1550e-9)
    print(f'propagator: (n_samples, window_t, dt, df): {propagator.n_samples}, {propagator.window_t}, {propagator.dt}, {propagator.df}')
    return laser_test_graph(**laser_params), propagator


@pytest.fixture(scope='session')
def SFL1550_graph_output():
    graph, propagator = get_SFL1550_graph_propagator()
    graph.propagate(propagator)
    return graph.measure_propagator(1)


@pytest.fixture(scope='session')
def SFL1550_graph_propagator():
    _, propagator = get_SFL1550_graph_propagator()
    return propagator


@pytest.fixture(scope='session')
def ORS1500_graph_output():
    graph, propagator = get_ORS1500_graph_propagator()
    graph.propagate(propagator)
    return graph.measure_propagator(1)


@pytest.fixture(scope='session')
def ORS1500_graph_propagator():
    _, propagator = get_ORS1500_graph_propagator()
    return propagator


@pytest.fixture(scope='session')
def ADJ_E15_graph_output():
    graph, propagator = get_ADJUSTIK_E15_graph_propagator()
    graph.propagate(propagator)
    return graph.measure_propagator(1)


@pytest.fixture(scope='session')
def ADJ_E15_graph_propagator():
    _, propagator = get_ADJUSTIK_E15_graph_propagator()
    return propagator


@pytest.fixture(scope='session')
def ADJ_X15_graph_output():
    graph, propagator = get_ADJUSTIK_E15_graph_propagator()
    graph.propagate(propagator)
    return graph.measure_propagator(1)


@pytest.fixture(scope='session')
def ADJ_X15_graph_propagator():
    _, propagator = get_ADJUSTIK_E15_graph_propagator()
    return propagator


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
@pytest.mark.CWL_validation
def test_SFL1550_lasing_spectrum(SFL1550_graph_output, SFL1550_graph_propagator):
    plot_psd_v_wavelength(SFL1550_graph_output, SFL1550_graph_propagator, title=f'SFL1550 Single Mode Lasing Spectrum (simulated), OSNR guessed', dB=True)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
@pytest.mark.CWL_validation
def test_SFL1550_lineshape(SFL1550_graph_output, SFL1550_graph_propagator):
    plot_lineshape_dB(SFL1550_graph_output, SFL1550_graph_propagator, (-1e4, 1e7), title='SFL1550 Linewidth Measurement (simulated)')


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
@pytest.mark.CWL_validation
def test_ORS1500_lineshape(ORS1500_graph_output, ORS1500_graph_propagator):
    plot_lineshape(ORS1500_graph_output, ORS1500_graph_propagator, (-2, 2), 0.5, title='ORS1500 Linewidth Measurement (simulated)')

@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
@pytest.mark.CWL_validation
def test_ADJUSTIK_phase_amplitude(ADJ_X15_graph_output, ADJ_E15_graph_output, ADJ_E15_graph_propagator):
    # note that the two adjustiks have the same propagator
    plot_phase_psd_v_freq(ADJ_E15_graph_output, ADJ_X15_graph_output, ADJ_E15_graph_propagator, (-10e3, 10e3), 0.1e3, title='ADJUSTIK PHASE v FREQUENCY')


# --------------------- EDFA Validation ------------------
def get_amp_graph(**kwargs):
    # no laser noise added, such that input power is exactly as wanted
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1e-6, 'central_wl': 1.55e-6, 'osnr_dB':1000, 'FWHM_linewidth':0}),
             1: OpticalAmplifier(parameters_from_name=kwargs),
             2: MeasurementDevice()
            }

    edges = [(0, 1),
             (1, 2)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


@pytest.fixture(scope='session')
def default_propagator():
    return Propagator(window_t=1e-9, n_samples=2**14, central_wl=1.55e-6)


@pytest.fixture(scope='function')
def EDFA100s_graph():
    # Gain flatness obtained by visual inspection of a typical gain graph is about 11 (across 100S) for small signal gain
    EDFA_params = {'max_small_signal_gain_dB':30, 'peak_wl':1550e-9, 'band_lower':1530e-9, 'band_upper':1565e-9, 'P_in_max':10e-3, 'P_out_max':100e-3, 'gain_flatness_dB':11, 'alpha':1, 'max_noise_fig_dB':5}
    return get_amp_graph(**EDFA_params)


def EDFA100p_graph(scope='function'):
    # Gain flatness obtained by visual inspection of a typical gain graph is about 11 (across 100P) for small signal gain
    EDFA_params = {'max_small_signal_gain_dB':28, 'peak_wl':1550e-9, 'band_lower':1530e-9, 'band_upper':1565e-9, 'P_in_max':10e-3, 'P_out_max':100e-3, 'gain_flatness_dB':11, 'alpha':1, 'max_noise_fig_dB':5}
    return get_amp_graph(**EDFA_params)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
@pytest.mark.EDFA_validation
def test_plot_input_to_output_power_100s(EDFA100s_graph, default_propagator):
    input_powers_dBm = np.linspace(-30, 7, num=50)
    input_powers = dBm_to_W(input_powers_dBm)
    output_powers = []

    for i in range(input_powers.shape[0]):
        input_power = input_powers[i]
        print(f'input power: {input_power}')
        EDFA100s_graph.nodes[0]['model'].parameters[0] = input_power # bit of a hack to get the desired input power
        EDFA100s_graph.nodes[0]['model'].set_parameters_as_attr()
        EDFA100s_graph.propagate(default_propagator)
        output = EDFA100s_graph.measure_propagator(2) # at measurement
        output_powers.append(np.mean(power_(output)))

    
    output_powers = np.array(output_powers)
    print(f'output powers: {output_powers}')

    _, ax = plt.subplots()
    # print(f'input_powers: {input_powers}')
    # print(f'output_powers: {output_powers}')
    ax.plot(W_to_dBm(input_powers), W_to_dBm(output_powers))
    ax.set_xlabel('Input power (dBm)')
    ax.set_ylabel('Output power (dBm)')
    plt.title('Typical EDFA 100S Output vs Input power (simulated)')
    plt.show()






