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


def test_incorrect_distribution():
    with pytest.raises(ValueError):
        AdditiveNoise(distribution='lorentzian')


def test_incorrect_noise_type():
    with pytest.raises(ValueError):
        AdditiveNoise(noise_type='transcendent')


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


def test_incompatible_signal_shapes(signal):
    noise = AdditiveNoise()
    noise.add_noise_to_propagation(signal)
    with pytest.raises(ValueError):
        noise.add_noise_to_propagation(signal[0:100])


# test autograd capabilities
def gain_sum_no_noise(signal):
    return np.sum(np.abs(5 * signal))

def gain_sum_noise(signal):
    gain = 5 * signal
    noise = AdditiveNoise(noise_param=4, seed=0)
    output = noise.add_noise_to_propagation(gain)
    return np.sum(np.abs(output))

def get_funct_gain_sum_noise(test_signal):
    noise = AdditiveNoise(noise_param=4, seed=0)
    noise.add_noise_to_propagation(test_signal) # try without this too...
                                                # But in practice if we evaluate before we derive
                                                # this test is accurate to what would happen
    def funct(signal):
        gain = 5 * signal
        output = noise.add_noise_to_propagation(gain)
        return np.sum(np.abs(output))
    
    return funct

@pytest.mark.skip
def test_autograd_gain1(signal):
    tmp = np.copy(signal)
    print(f'Sum power no noise: {gain_sum_no_noise(tmp)}')
    
    tmp = np.copy(signal)
    print(f'Sum power noise: {gain_sum_noise(tmp)}')
    
    tmp = np.copy(signal)
    gain_separate_noise_gen = get_funct_gain_sum_noise(tmp)
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
    gain_separate_noise_gen = get_funct_gain_sum_noise(tmp)
    tmp = np.copy(signal)
    print(f'grad sum power separate noise: {grad_separate_noise_gen(tmp)}')

    assert False

@pytest.mark.skip
def test_autograd_convergence(signal):
    # same as the gain test, but we'll take the average of a BUNCH of noisy signals and see if it converges
    TOTAL_ITERATIONS = 1

    grad_sum = np.zeros((signal.shape[0], 1), dtype='complex')
    noise = AdditiveNoise(noise_param=4, seed=0, noise_on=True, noise_type='absolute')
    noise.add_noise_to_propagation(np.copy(signal))

    for i in range(TOTAL_ITERATIONS):
        tmp = np.copy(signal)
        noise.resample_noise(seed=i)
        
        def funct(x):
            """
            Making this a linear function bc it's easier to validate the derivative there
            """
            gain = 5 * x
            output = noise.add_noise_to_propagation(gain)
            return np.sum(output).astype(dtype='float')
        
        grad_funct = grad(funct)
        print("About to evaluate grad function!")
        grad_sum += grad_funct(tmp)

    avg_grad = grad_sum / TOTAL_ITERATIONS
    print("Average gradient:")
    print(avg_grad)
    assert False

def test_autograd_gain_signal_absolute():
    signal = np.zeros((10000, 1), dtype='complex')
    noise = AdditiveNoise(noise_param=4, seed=0, noise_on=True, noise_type='absolute')
    noise.add_noise_to_propagation(signal)

    noise.resample_noise(seed=0)
    def funct(G):
        all_noise = noise.add_noise_to_propagation(signal) # using empty signal but doesn't matter, it's a constant with this deriv
        print(f"all_noise sum: {np.sum(np.abs(all_noise))}")
        return np.sum(np.abs(G * all_noise))
    
    grad_funct = grad(funct)
    print("Evaluating grad function")
    gradient_at_G = grad_funct(np.array([5]))
    print(gradient_at_G)

    # get noise sum
    noise.resample_noise(seed=0)
    noise_sum = np.sum(np.abs(noise.add_noise_to_propagation(signal)))
    assert np.isclose(gradient_at_G[0], noise_sum)

def test_autograd_gain_signal_relative():
    GAIN = 8

    signal = np.ones((10000, 1), dtype='complex')
    noise = AdditiveNoise(noise_param=4, seed=0, noise_on=True, noise_type='relative')
    noise.add_noise_to_propagation(signal)

    noise.resample_noise(seed=0)
    def funct(G):
        output = noise.add_noise_to_propagation(G * signal) -  G * signal
        return np.sum(output).astype(dtype='float')
    
    grad_funct = grad(funct)
    print("Evaluating grad function")
    gradient_at_G = grad_funct(np.array([GAIN]))
    print(gradient_at_G)

    # test with expected val
    noise.resample_noise(seed=0)
    noise_sum = np.sum(noise.add_noise_to_propagation(signal) - signal).astype(dtype='float')
    print(f"noise_sum: {noise_sum}")
    assert np.isclose(gradient_at_G[0], noise_sum)
