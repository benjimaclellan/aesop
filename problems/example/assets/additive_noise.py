import autograd.numpy as np
import matplotlib.pyplot as plt

from .functions import fft_, power_, psd_

"""
TODO: Check with Ben or Piotr
- Do we want the noise param to match the matlab script (i.e. be in dB) or to be the ratio (am leaning to the Matlab bc the scaling might be more sense)
TODO: update functions to take propagator (not just vector) 1. Update functions 2. Update tests 3. change the frequency representation to be normalized smh
"""


class AdditiveNoise():
    """
    Additive noise is noise modelled as: <total signal>(t) = <pure signal>(t) + <noise>(t)

    This class allows for the shape of noise to be defined, and added to signal states
    """

    simulate_with_noise = True

    def __init__(self, distribution='gaussian', noise_param=1, noise_filter=None, noise_type='relative', noise_on=True, seed=None):
        """
        Creates an noise object with the described properties

        :param sample_num: number of noise samples to provide (should match sample number of signals to which noise is applied)
        :param distribution: type of noise distribution. ONLY GAUSSIAN IS SUPPORTED AT THIS TIME
        :param noise_param: scaling parameter for noise (standard deviation for Gaussian)
        :param noise_filter: None for now
        :param noise_type: 'relative' or 'absolute'. If relative, noise is scaled 
        :param noise_on: True if the noise of the object is to be applied, False otherwise
                         Note that the noise can also be turned off for the whole class (upon which this variable has no effect)
        :param seed: seed with which np.random can generate the pseudorandom noise vectors

        :raises ValueError if (1) non-gaussian distribution is requested (2) someone has the audacity of asking for a filter I have yet to implement
                              (3) noise_type is not 'relative' or 'absolute'
        """
        self._sample_num = None # defined the first time that noise is added to the propagation
    
        if (noise_filter is not None):
            raise ValueError('Noise filter options not yet implemented, object must be None')
        
        self._seed = seed # can't actually seed it now because we have no idea what random calls will come between initialization and the first call to add_noise_to_propagate
    
        # keep log of noise sources for debugging, and in case of resampling
        self.noise_sources = []

        # create noise distribution
        self._sample_num = None
        self.add_noise_source(distribution=distribution, noise_param=noise_param, noise_filter=noise_filter, noise_type=noise_type)

        self.noise_on = noise_on
    
    def add_noise_source(self, distribution='gaussian', noise_param=1, noise_filter=None, noise_type='relative'):
        """
        Adds a noise source to the AdditiveNoise object (all future calls to add_noise_to_propagation will also include this source)
        WARNING: all noise sources should probably be added at once (and non-concurrently please, this is not threadsafe yet) so that seeding actually makes results predictable
                 We will seed ONCE when initializing the noise sources, so if you have an external random call, might make it harder to track down the precise noise vectors 

        :param distribution: type of noise distribution. ONLY GAUSSIAN IS SUPPORTED AT THIS TIME
        :param noise_param: scaling parameter for noise (standard deviation for Gaussian)
        :param noise_filter: None for now
        :param noise_type: 'relative' or 'absolute'. If relative, noise is scaled
        :param seed: seed with which np.random can generate the pseudorandom noise vectors
    
        :raises ValueError if (1) non-gaussian distribution is requested (2) someone has the audacity of asking for a filter I have yet to implement
                              (3) noise_type is not 'relative' or 'absolute'
        """
        if (distribution != 'gaussian'):
            raise ValueError(f"{distribution} distribution not supported. Only 'gaussian' is supported at this time")
        
        if (noise_type != 'relative' and noise_type != 'absolute'):
            raise ValueError('Noise type must be relative or absolute')
        
        if (noise_filter is not None):
            raise ValueError('Noise filter options not yet implemented, object must be None')

        if (self._sample_num is not None):
            noise = np.random.normal(scale=1, size=(self.sample_num, 2)).view(dtype='complex').flatten()
            noise_power_norm = np.linalg.norm(power_(noise))**0.5
        else:
            noise = None
            noise_power_norm = None

        source = {'distribution': distribution,
                  'noise_param': noise_param,
                  'filter' : noise_filter,
                  'noise_type': noise_type,
                  'noise_vector': noise,
                  'noise_power_norm': noise_power_norm
                 }
        self.noise_sources.append(source)
    
    def add_noise_to_propagation(self, signal):
        """
        Adds noise to the input state, and returns the noisy state
        
        :param signal: the pre-noise state
        :return: the state including noise
        """
        if (self._sample_num is None):
            self.sample_num = signal.shape[0]
        elif (self.sample_num != signal.shape[0]):
            raise ValueError(f'signal length is {signal.shape[0]} much match sample_num {self.sample_num}')

        if (not self.noise_on or not AdditiveNoise.simulate_with_noise):
            return signal
        
        signal_power_norm = np.linalg.norm(power_(signal))**0.5 #TODO: Think about... for now it just matches Matlab

        for noise in self.noise_sources:
            #TODO: figure out why it's divided by 20 and not 10 in that power of 10...
            if noise['noise_type'] == 'relative':
                signal += noise['noise_vector'] * signal_power_norm / noise['noise_power_norm'] * 10**(-1 * noise['noise_param'] / 20)
            else:
                signal += noise['noise_vector'] * 10**(noise['noise_param'] / 20)

        print(f"Noise power norm unscaled: {noise['noise_power_norm']}")
        print(f"Signal power norm pure: {signal_power_norm}")


        return signal

    def display_noisy_signal(self, signal):
        """
        Displays plots the noisy signal power in the time and frequency domains

        Note that signal WILL set sample_num value
        """
        #TODO: fix up the axes a bit but tbh I just want to see the shape
        _, ax = plt.subplots(2, 1)

        noisy_signal = self.add_noise_to_propagation(signal) 
        x = np.linspace(-1, 1, num=self.sample_num)

        ax[0].plot(x, power_(noisy_signal), label='time domain')
        ax[1].plot(x, 10 * np.log10(psd_(noisy_signal, 1, 1)), label='freq domain')
        ax[0].legend()
        ax[1].legend()
        plt.title('Noisy signal')
        plt.show()
    
    def display_noise(self, signal):
        """
        Displays plots the noise power in the time and frequency domains

        Note that signal WILL set sample_num value
        """
        #TODO: fix up the axes a bit but tbh I just want to see the shape
        _, ax = plt.subplots(2, 1)

        pure_signal = np.copy(signal)
        noise_vector = self.add_noise_to_propagation(signal) - pure_signal
        x = np.linspace(-1, 1, num=self.sample_num)

        ax[0].plot(x, power_(noise_vector), label='time domain')
        ax[1].plot(x, 10 * np.log10(psd_(noise_vector, 1, 1)), label='freq domain (log)')
        ax[0].legend()
        ax[1].legend()
        plt.title('Total noise')
        plt.show()

    def display_noise_sources_absolute(self):
        """
        Displays plots the power of individual noise contributions in the time and frequency domains

        Since no input vector is provided, outputs are displayed as if all noise source options were absolute

        If self._sample_num is None, will display plot for 1000 datapoints
        """
        #TODO: fix up the axes a bit but tbh I just want to see the shape

        display_only_sample = False
        if (self._sample_num is None):
            self.sample_num = 1000
            display_only_sample = True
    
        _, ax = plt.subplots(2, 1)

        x = np.linspace(-1, 1, num=self.sample_num)
        for noise in self.noise_sources:
            noise_param = noise['noise_param']
            noise_vector = noise['noise_vector'] * noise_param
            ax[0].plot(x, power_(noise_vector), label=f'time domain, param: {noise_param}')
            ax[1].plot(x, psd_(noise_vector, 1, 1), label=f'freq domain, param: {noise_param}')
        
        if (display_only_sample): # we only want to remove the sample number if we set it ourselves at the top of the method
            self._sample_num = None

        ax[0].legend()
        ax[1].legend()
        plt.title('Display each noise source as absolute')
        plt.show()
    
    @property
    def sample_num(self):
        if (self._sample_num is None):
            raise ValueError('No sample num set yet')
        return self._sample_num
    
    @sample_num.setter
    def sample_num(self, n):
        if (self._seed is not None):
            np.random.seed(self._seed) # things will be replicable (if single threaded)

        for noise_source in self.noise_sources:
            noise = np.random.normal(scale=1, size=(n, 2)).view(dtype='complex').flatten()
            noise_power_norm = np.linalg.norm(power_(noise))**0.5
            noise_source['noise_vector'] = noise
            noise_source['noise_power_norm'] = noise_power_norm
        
        self._sample_num = n


def testing_main():
    noise = AdditiveNoise(noise_type='absolute', noise_param=4)
    noise.display_noise()
    noise.add_noise_source(noise_param=2) # relative noises
    noise.add_noise_source(noise_param=1)
    noise.add_noise_source(noise_param=0.5)
    noise.display_noise_sources_absolute()
    noise.noise_on = False
    noise.display_noise()
    noise.noise_on = True
    noise.add_noise_to_propagation(np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype='complex'))
    noise.display_noise()
    AdditiveNoise.simulate_with_noise = False
    noise.display_noise()
    noise.display_noise_sources_absolute()
