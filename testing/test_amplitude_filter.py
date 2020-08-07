import autograd.numpy as np
import pytest

from problems.example.assets.propagator import Propagator
from problems.example.assets.filter import AmplitudeFilter


SKIP_GRAPHICAL_TEST = True


@pytest.fixture(scope='function')
def propagator():
    return Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)


@pytest.fixture(scope='function')
def freq_squared_filter():
    return AmplitudeFilter(central_wl=1550e-9, FWHM=50e-9)


@pytest.fixture(scope='function')
def noise_signal(propagator):
    return np.random.normal(scale=1, size=(propagator.n_samples, 2)).view(dtype='complex')


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_freq_squared_filter(freq_squared_filter, propagator):
    freq_squared_filter.display_filter(propagator)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_freq_squared_filtered_output(freq_squared_filter, propagator, noise_signal):
    freq_squared_filter.display_filtered_output(noise_signal, propagator)
