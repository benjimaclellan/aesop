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
def test_standard_butterworth(butterworth_lowpass, propagator):
    butterworth_lowpass.display_filter(propagator)


def test_multiple_orders_butterworth(propagator):
    """
    Amplitude validated by inspection
    Phase validated by comparing to matlab plots obtained like so: https://www.mathworks.com/help/signal/ref/butter.html
    """
    _, ax = plt.subplots(2)

    for i in range(4, 8):# [1, 2, 5, 8, 11, 14]:
        filter_shape = Filter.get_filter(propagator, shape='butterworth lowpass', transition_f=100e9, dc_gain=1, order=i)
        freq = ifft_shift_(propagator.f) # we'll only be looking at positive frequencies, so might as well put them first
        ax[0].plot(freq[0:2000], np.abs(filter_shape[0:2000]), label=f'order: {i}')
        ax[0].set_xlabel('frequency (Hz)')
        ax[0].set_ylabel('Filter amplitude (dB)')
        ax[1].plot(freq[0:2000], np.angle(filter_shape[0:2000]), label=f'order: {i}')
        ax[1].set_xlabel('frequency (Hz)')
        ax[1].set_ylabel('Filter phase (rad)')
    
    ax[0].legend()
    ax[1].legend()
    plt.title('Butterworth filter')
    plt.show()


def test_filtering_butterworth():
    pass


def test_filtering_butterworth_multiple_orders():
    pass
