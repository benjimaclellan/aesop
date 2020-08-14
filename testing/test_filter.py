import autograd.numpy as np
import pytest
import matplotlib.pyplot as plt

from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import ifft_shift_
from problems.example.assets.filter import Filter


SKIP_GRAPHICAL_TEST = True


@pytest.fixture(scope='function')
def propagator():
    return Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)


@pytest.fixture(scope='function')
def freq_squared_filter():
    return Filter(shape='exponential of square', central_wl=1550e-9, FWHM=50e-9)


@pytest.fixture(scope='function')
def butterworth_lowpass():
    return Filter(shape='butterworth lowpass', transition_f=100e9, dc_gain=1, order=2)


@pytest.fixture(scope='function')
def noise_signal(propagator):
    return np.random.normal(scale=1, size=(propagator.n_samples, 2)).view(dtype='complex')


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_freq_squared_filter(freq_squared_filter, propagator):
    freq_squared_filter.display_filter(propagator)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_freq_squared_filtered_output(freq_squared_filter, propagator, noise_signal):
    freq_squared_filter.display_filtered_output(noise_signal, propagator)

# ------------------ TEST BUTTERWORTH ---------------------------
@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_standard_butterworth(butterworth_lowpass, propagator):
    butterworth_lowpass.display_filter(propagator)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_multiple_orders_butterworth(propagator):
    """
    Amplitude validated by inspection
    Phase validated by comparing to matlab plots obtained like so: https://www.mathworks.com/help/signal/ref/butter.html
    """
    _, ax = plt.subplots(2)

    for i in [1, 2, 5, 8]:
        filter_shape = Filter.get_filter(propagator, shape='butterworth lowpass', transition_f=100e9, dc_gain=1, order=i)
        freq = ifft_shift_(propagator.f) # we'll only be looking at positive frequencies, so might as well put them first
        ax[0].plot(freq[0:1000], np.abs(filter_shape[0:1000]), label=f'order: {i}')
        ax[0].set_xlabel('frequency (Hz)')
        ax[0].set_ylabel('Filter amplitude (dB)')
        ax[1].plot(freq[0:1000], np.angle(filter_shape[0:1000]), label=f'order: {i}')
        ax[1].set_xlabel('frequency (Hz)')
        ax[1].set_ylabel('Filter phase (rad)')
    
    ax[0].legend()
    ax[1].legend()
    plt.title('Butterworth filter')
    plt.show()


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_filtering_butterworth(propagator, butterworth_lowpass):
    signal_dc = np.ones_like(propagator.t)
    signal_low = np.cos(2 * np.pi * 10e9 * propagator.t)
    signal_high = np.cos(2 * np.pi * 150e9 * propagator.t)
    signal_highest = np.cos(2 * np.pi * 300e9 * propagator.t)
    signal = signal_dc + signal_low + signal_high + signal_highest

    _, ax = plt.subplots(2, 1)
    ax[0].plot(propagator.t, signal_dc + 10, label='DC (offset 10)')
    ax[0].plot(propagator.t, signal_low + 10, label='low freq (offset 10)')
    ax[0].plot(propagator.t, signal_high + 10, label='high freq (offset 10)')
    ax[0].plot(propagator.t, signal_highest + 10, label='highest freq (offset 10)', lw=1, ls=':')

    ax[0].plot(propagator.t, signal, label='signal', lw=1)
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('Signal unfiltered')
    ax[0].legend()

    ax[1].plot(propagator.t, butterworth_lowpass.get_filtered_time(signal_dc, propagator) + 10, label='DC (offset 10)')
    ax[1].plot(propagator.t, butterworth_lowpass.get_filtered_time(signal_low, propagator) + 10, label='low freq (offset 10)')
    ax[1].plot(propagator.t, butterworth_lowpass.get_filtered_time(signal_high, propagator) + 10, label='high freq (offset 10)')
    ax[1].plot(propagator.t, butterworth_lowpass.get_filtered_time(signal_highest, propagator) + 10, label='highest freq (offset 10)', lw=1, ls=':')
    ax[1].plot(propagator.t, butterworth_lowpass.get_filtered_time(signal, propagator), label='signal', lw=1)
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('Signal filtered')
    ax[1].legend()

    plt.title('Butterworth: filtered vs unfiltered signals')
    plt.show()


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_filtering_butterworth_multiple_orders(propagator):
    signal = 2 * np.cos(2 * np.pi * 50e9 * propagator.t) + np.cos(2 * np.pi * 150e9 * propagator.t)

    NUM_FILTERS = 4

    _, ax = plt.subplots(NUM_FILTERS + 1, 1)

    ax[0].plot(propagator.t, signal, label='signal', lw=1)
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('Unfiltered')
    ax[0].legend()

    for i in range(1, NUM_FILTERS + 1):
        butter = Filter(shape='butterworth lowpass', transition_f=100e9, dc_gain=1, order=i)
        ax[i].plot(propagator.t, butter.get_filtered_time(signal, propagator), lw=1)
        ax[i].set_xlabel('time (s)')
        ax[i].set_ylabel(f'order {i}')
        ax[i].legend()

    plt.show()