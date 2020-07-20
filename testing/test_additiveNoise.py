import autograd.numpy as np
import pytest
from autograd import grad

from problems.example.assets.additive_noise import AdditiveNoise
from problems.example.assets.propagator import Propagator

@pytest.fixture(autouse=True)
def setup():
    AdditiveNoise.simulate_with_noise = True


def display_time_freq(noise):
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    signal = np.sin(2 * np.pi / 1e-9 * propagator.t) + 1j*0 # we just really want a visible signal that's not constant power so *shrugs*
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
def test_display_timeFreq():
    # noise param = 1
    noise1 = AdditiveNoise(noise_param=1, seed=0)
    display_time_freq(noise1)

    # noise param = 5
    noise5 = AdditiveNoise(noise_param=5, seed=0)
    display_time_freq(noise5)

    # noise param = 10
    noise10 = AdditiveNoise(noise_param=10, seed=0)
    display_time_freq(noise10)

    # noise param = 32
    noise32 = AdditiveNoise(noise_param=32, seed=0)
    display_time_freq(noise32)

# test multiple noise sources in one
def test_multiple_noise_sources():
    pass

# test absolute vs relative (with new object)

# test incorrect noise type

# test add_noise_to_propagation
    
# test that noise arrays are not recomputed everytime add_noise_to_propagation is added

# test noise_on and simulate with noise

# test incompatible shapes

# test autograd capabilities