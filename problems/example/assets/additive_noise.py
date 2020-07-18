import autograd.numpy as np
import matplotlib.pyplot as plt

from .functions import fft_, power_, psd_

"""
TODO: Check with Ben or Piotr
- Does it make sense to have the noise type be either relative or absolute? I think so
    - Generally higher power = more noise
    - But there could be an environmental source or something which would not be dependent of power
- Do we want the noise param to match the matlab script (i.e. be in dB) or to be the ratio (am leaning to the Matlab bc the scaling might be more sense)
- OK so technically specifying the sample_num here is a problem bc it restricts the propagator side. SO THIS CANNOT STAY.
    That said, regenerating all this info at each function call is extra
    Think about a neat solution to this design flaw
"""


class AdditiveNoise():
    """
    Additive noise is noise modelled as: <total signal>(t) = <pure signal>(t) + <noise>(t)

    This class allows for the shape of noise to be defined, and added to signal states
    """

    simulate_with_noise = True

    def __init__(self, sample_num=2**14, distribution='gaussian', noise_param=1, noise_filter=None, noise_type='relative', noise_on=True, seed=None):
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
        """
        self.sample_num = sample_num
        
        if (distribution != 'gaussian'):
            raise ValueError(f"{distribution} distribution not supported. Only 'gaussian' is supported at this time")
        if (noise_filter is not None):
            raise ValueError('Noise filter options not yet implemented, object must be None')
        if (noise_type != 'relative' and noise_type != 'absolute'):
            raise ValueError('Noise type must be relative or absolute')
        
        # keep log of noise sources for debugging, and in case of resampling
        self.noise_sources = []

        # create noise distribution
        self.sample_num = sample_num
        self.add_noise_source(distribution=distribution, noise_param=noise_param, noise_filter=noise_filter, noise_type=noise_type, seed=seed)

        self.noise_on = noise_on
    
    def add_noise_source(self, distribution='gaussian', noise_param=1, noise_filter=None, noise_type='relative', seed=None):
        """
        Adds a noise source to the AdditiveNoise object (all future calls to add_noise_to_propagation will also include this source)

        :param distribution: type of noise distribution. ONLY GAUSSIAN IS SUPPORTED AT THIS TIME
        :param noise_param: scaling parameter for noise (standard deviation for Gaussian)
        :param noise_filter: None for now
        :param noise_type: 'relative' or 'absolute'. If relative, noise is scaled
        :param seed: seed with which np.random can generate the pseudorandom noise vectors
        """
        if (distribution != 'gaussian'):
            raise ValueError(f"{distribution} distribution not supported. Only 'gaussian' is supported at this time")
        if (seed is not None):
            np.random.seed(seed)

        noise = np.random.normal(scale=1, size=(self.sample_num, 2)).view(dtype='complex').flatten()
        noise_scaling = np.linalg.norm(power_(noise))**0.5
        source = {'distribution': distribution,
                  'noise_param': noise_param,
                  'filter' : noise_filter,
                  'noise_type': noise_type,
                  'noise_vector': noise,
                  'noise_scaling': noise_scaling
                 }
        self.noise_sources.append(source)
    
    def add_noise_to_propagation(self, state):
        """
        Adds noise to the input state, and returns the noisy state
        
        :param state: the pre-noise state
        :return: the state including noise
        """
        if (not self.noise_on or not AdditiveNoise.simulate_with_noise):
            return state
        
        state_scaling = np.linalg.norm(power_(state))**0.5 #TODO: Think about... for now it just matches Matlab

        for noise in self.noise_sources:
            #TODO: figure out why it's divided by 20 and not 10 in that power of 10...
            if noise['noise_type'] == 'relative':
                state += noise['noise_vector'] * state_scaling / noise['noise_scaling'] * 10**(noise['noise_param'] / 20)
            else:
                state += noise['noise_vector'] * 10**(noise['noise_param'] / 20)

        return state

    def display_noise(self):
        """
        Displays plots the noise power in the time and frequency domains
        """
        #TODO: fix up the axes a bit but tbh I just want to see the shape
        _, ax = plt.subplots(2, 1)

        x = np.linspace(-1, 1, num=self.sample_num)
        noise_vector = self.add_noise_to_propagation(np.zeros(self.sample_num, dtype='complex'))
        ax[0].plot(x, power_(noise_vector), label='time domain')
        ax[1].plot(x, psd_(noise_vector, 1, 1), label='freq domain')
        ax[0].legend()
        ax[1].legend()
        plt.show()
    
    def display_noise_sources(self):
        """
        Displays plots the power of individual noise contributions in the time and frequency domains

        Since no input vector is provided, outputs are displayed as if all noise source options were absolute
        """
        #TODO: fix up the axes a bit but tbh I just want to see the shape
        _, ax = plt.subplots(2, 1)

        x = np.linspace(-1, 1, num=self.sample_num)
        for noise in self.noise_sources:
            noise_param = noise['noise_param']
            noise_vector = noise['noise_vector'] * noise_param
            ax[0].plot(x, power_(noise_vector), label=f'time domain, param: {noise_param}')
            ax[1].plot(x, psd_(noise_vector, 1, 1), label=f'freq domain, param: {noise_param}')

        ax[0].legend()
        ax[1].legend()
        plt.show()

def testing_main():
    noise = AdditiveNoise(sample_num=10000, noise_type='absolute', noise_param=4)
    noise.display_noise()
    noise.add_noise_source(noise_param=2)
    noise.add_noise_source(noise_param=1)
    noise.add_noise_source(noise_param=0.5)
    noise.display_noise_sources()
    noise.noise_on = False
    noise.display_noise()
    noise.noise_on = True
    noise.display_noise()
    AdditiveNoise.simulate_with_noise = False
    noise.display_noise()
    noise.display_noise_sources()
