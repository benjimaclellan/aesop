import autograd.numpy as np
import pytest
from autograd import grad

from problems.example.assets.additive_noise import AdditiveNoise
from problems.example.assets.propagator import Propagator

@pytest.fixture(autouse=True)
def setup():
    AdditiveNoise.simulate_with_noise = True


@pytest.fixture(scope='function')
def propagator():
    return Propagator(window_t=1e-9, n_samples=2**14, central_wl=1.55e-6)


@pytest.fixture(scope='function')
def signal(propagator):
    return np.sin(2 * np.pi / 1e-9 * propagator.t) + 1j*0 # we just really want a visible signal that's not constant power so *shrugs*


def display_time_freq(noise, signal, propagator):
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    noise.display_noisy_signal(signal, propagator=propagator)
    noise.display_noise(signal, propagator=propagator)
    noise.display_noise_sources_absolute(propagator=propagator)


# test not Gaussian and incorrect noise type
def test_incorrect_distribution():
    with pytest.raises(ValueError):
        AdditiveNoise(distribution='lorentzian')


def test_incorrect_noise_type():
    with pytest.raises(ValueError):
        AdditiveNoise(noise_type='transcendent')


# simple test, visually assess whether the time-freq domain look correct
@pytest.mark.skip
def test_display_timeFreq(signal, propagator):
    # noise param = 1
    noise1 = AdditiveNoise(noise_param=1, seed=0)
    display_time_freq(noise1, signal, propagator)

    # noise param = 5
    noise5 = AdditiveNoise(noise_param=5, seed=0)
    display_time_freq(noise5, signal, propagator)

    # noise param = 10
    noise10 = AdditiveNoise(noise_param=10, seed=0)
    display_time_freq(noise10, signal, propagator)

    # noise param = 32
    noise32 = AdditiveNoise(noise_param=32, seed=0)
    display_time_freq(noise32, signal, propagator)


# test multiple noise sources in one
@pytest.mark.skip
def test_multiple_noise_sources(signal, propagator):
    noise = AdditiveNoise(noise_param=2, seed=0)
    noise.display_noisy_signal(signal, propagator=propagator)
    noise.display_noise(signal, propagator=propagator)

    noise.add_noise_source(noise_param=4) # relative noises
    noise.add_noise_source(noise_param=8)
    noise.add_noise_source(noise_param=16)
    
    noise.display_noisy_signal(signal, propagator=propagator)
    noise.display_noise(signal, propagator=propagator)
    noise.display_noise_sources_absolute(propagator=propagator)


# test absolute vs relative (with new object)
@pytest.mark.skip
def test_absolute_vs_relative(signal, propagator):
    signal2 = 2 * np.copy(signal)

    # relative: test against two input vectors and confirm magnitude scales
    noise = AdditiveNoise(noise_param=2, seed=0)
    noise.display_noise(signal, propagator=propagator)
    noise = AdditiveNoise(noise_param=2, seed=0)
    noise.display_noise(signal2, propagator=propagator) # noise expected to have 4x larger amplitude

    # absolute: test against two input vectors and confirm magnitude doesn't change
    noise = AdditiveNoise(noise_param=2, seed=0, noise_type='absolute')
    noise.display_noise(signal, propagator=propagator)
    noise = AdditiveNoise(noise_param=2, seed=0, noise_type='absolute')
    noise.display_noise(signal2, propagator=propagator) # noise expected to have same amplitude


# test noise_on and simulate with noise
@pytest.mark.skip
def test_noise_disactivation(signal, propagator):
    noise = AdditiveNoise(noise_param=4, seed=0)
    noise.display_noisy_signal(signal, propagator=propagator) # should show pure signal
    noise.display_noise(signal, propagator=propagator)
    noise.noise_on = False
    noise.display_noise(signal, propagator=propagator) # should show nothing
    noise.noise_on = True
    noise.display_noise(signal, propagator=propagator) # should be back to the start
    AdditiveNoise.simulate_with_noise = False
    noise.display_noisy_signal(signal, propagator=propagator) # should show pure signal
    noise.display_noise(signal, propagator=propagator) # should show nothing


# test incompatible shapes
def test_incompatible_signal_shapes(signal):
    noise = AdditiveNoise()
    noise.add_noise_to_propagation(signal)
    with pytest.raises(ValueError):
        noise.add_noise_to_propagation(signal[0:100])


# test autograd capabilities

# test that noise arrays are not recomputed everytime add_noise_to_propagation is added
