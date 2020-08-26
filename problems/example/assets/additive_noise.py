import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.constants import elementary_charge

from .functions import fft_, ifft_, power_, psd_, ifft_shift_

"""
TODO: change frequencyrep to be normalized like the matlab code
TODO: can we create different copies of the class for threadsafe-ness purposes? Think about potential parallelization (one class per graph?)
"""

class AdditiveNoise():
    """
    Additive noise is noise modelled as: <total signal>(t) = <pure signal>(t) + <noise>(t)

    This class allows for the shape of noise to be defined, and added to signal states
    """

    simulate_with_noise = True

    def __init__(self, distribution='gaussian', noise_param=1, noise_filter=None, noise_type='osnr', edfa=None, noise_on=True, seed=None):
        """
        Creates an noise object with the described properties

        :param sample_num: number of noise samples to provide (should match sample number of signals to which noise is applied)
        :param distribution: type of noise distribution. ONLY GAUSSIAN IS SUPPORTED AT THIS TIME
        :param noise_param: scaling parameter for noise (if noise_type='osnr', this parameter is OSNR in dB. If 'absolute power', this param is average noise power)
        :param noise_filter: None for now
        :param noise_type: 'osnr', 'absolute power', or 'edfa ASE'
        :param noise_on: True if the noise of the object is to be applied, False otherwise
                         Note that the noise can also be turned off for the whole class (upon which this variable has no effect)
        :param seed: seed with which np.random can generate the pseudorandom noise vectors

        :raises ValueError if (1) non-gaussian distribution is requested (2) someone has the audacity of asking for a filter I have yet to implement
                              (3) noise_type is not 'osnr', 'absolute power', 'edfa ASE', 'rms constant', 'shot'
        """
        self._propagator = None # defined the first time that noise is added to the propagation
        
        self._seed = seed # can't actually seed it now because we have no idea what random calls will come between initialization and the first call to add_noise_to_propagate
    
        # keep log of noise sources for debugging, and in case of resampling
        self.noise_sources = []

        # create noise distribution
        self.add_noise_source(distribution=distribution, noise_param=noise_param, noise_filter=noise_filter, noise_type=noise_type, edfa=edfa)

        self.noise_on = noise_on
    
    def add_noise_source(self, distribution='gaussian', noise_param=1, noise_filter=None, noise_type='osnr', edfa=None):
        """
        Adds a noise source to the AdditiveNoise object (all future calls to add_noise_to_propagation will also include this source)
        WARNING: all noise sources should probably be added at once (and non-concurrently please, this is not threadsafe yet) so that seeding actually makes results predictable
                 We will seed ONCE when initializing the noise sources, so if you have an external random call, might make it harder to track down the precise noise vectors 

        :param distribution: type of noise distribution. ONLY GAUSSIAN IS SUPPORTED AT THIS TIME
        :param noise_param: scaling parameter for noise (if noise_type='osnr', this parameter is OSNR in dB. If 'absolute power', this is the total power of noise. 
                            If noise_type 'edfa ASE' in internal EDFA params are used
        :param noise_filter: None for now
        :param noise_type: 'osnr', 'absolute power', 'edfa ASE', 'rms constant', 'shot'
        :param seed: seed with which np.random can generate the pseudorandom noise vectors
    
        :raises ValueError if (1) non-gaussian distribution is requested (2) someone has the audacity of asking for a filter I have yet to implement
                              (3) noise_type is not 'osnr' or 'absolute power' or 'edfa ASE' or 'rms constant' or 'shot'
        """
        if (distribution != 'gaussian'):
            raise ValueError(f"{distribution} distribution not supported. Only 'gaussian' is supported at this time")
        
        if (noise_type != 'osnr' and noise_type != 'absolute power' and noise_type != 'edfa ASE' and \
            noise_type != 'rms constant' and noise_type != 'shot' and noise_type != 'FWHM linewidth'):
            raise ValueError(f'Noise type must be osnr, absolute power, edfa ASE, rms constant, shot. {noise_type} is not valid')

        if (noise_type == 'shot' and noise_filter is None):
            raise ValueError(f'Shot noise requires noise filter')

        if (self.propagator is not None):
            noise = np.random.normal(scale=1, size=(self.propagator.n_samples, 2)).view(dtype='complex')
            mean_noise_power = np.mean(power_(noise))
            if (noise_type == 'absolute power'):
                noise_scaling = np.sqrt(noise_param / mean_noise_power) # set average power to equal noise_param
                noise = noise * noise_scaling
            if (noise_type == 'rms constant'):
                noise_scaling = noise_param / np.sqrt(mean_noise_power) # technically mean noise power here is voltage squared but whatever, it works out mathematically
                noise = noise * noise_scaling
            if (noise_type == 'FWHM linewidth'):
                expected_phase_noise_amplitude = np.sqrt((noise_param / np.pi) / 2) / ifft_shift_(np.abs(self.propagator.f))
                phase_noise = expected_phase_noise_amplitude * AdditiveNoise._get_real_noise_signal_freq(self.propagator)
                noise = ifft_(phase_noise, self.propagator.dt)
        else:
            noise = None
            mean_noise_power = None
        
        if (noise_type == 'osnr'):
            noise_param = 10**(noise_param / 10) # convert OSNR(dB) to OSNR(ratio)

        source = {'distribution': distribution,
                  'noise_param': noise_param,
                  'filter' : noise_filter,
                  'noise_type': noise_type,
                  'noise_vector': noise,
                  'mean_noise_power': mean_noise_power,
                  'filter': noise_filter
                 }
        if (noise_type == 'edfa ASE'):
            source['edfa'] = edfa

        self.noise_sources.append(source)
    
    def add_noise_to_propagation(self, signal, propagator):
        """
        Adds noise to the input state, and returns the noisy state
        
        :param signal: the pre-noise state (IS MODIFIED BY THE FUNCTION)
        :return: the state including noise
        """   
        if (not self.noise_on or not AdditiveNoise.simulate_with_noise):
            return signal
   
        if (self.propagator is None or propagator is not self.propagator): # we avoid resampling the noise when the propagator hasn't changed, for efficiency
            self.propagator = propagator

        total_noise = np.zeros(signal.shape, dtype='complex')
        mean_signal_power = np.mean(power_(signal))

        for noise in self.noise_sources:
            if noise['noise_type'] == 'osnr':
                scaling_factor = (mean_signal_power / noise['mean_noise_power'] / noise['noise_param'])**0.5
                total_noise = total_noise + noise['noise_vector'] * scaling_factor
            elif noise['noise_type'] == 'absolute power' or noise['noise_type'] == 'rms constant':
                total_noise = total_noise + noise['noise_vector'] # noise vector already scaled according to noise param
            elif noise['noise_type'] == 'shot': # v_rms^2 = 2qBR^2(I_d + I_p), I_rms^2 = 2qB(I_d + I_p)
                target_squared_rms = (noise['noise_param'][0] * np.mean(np.abs(signal)) + noise['noise_param'][1]) * np.sum(np.abs(noise['filter'].filter)) * propagator.df
                scaling_factor = np.sqrt(target_squared_rms / np.mean(power_(noise['noise_vector'])))

                total_noise = total_noise + scaling_factor * noise['noise_vector']
            elif noise['noise_type'] == 'edfa ASE':
                ASE_expected_amplitude = noise['edfa'].get_ASE_filter(propagator) * np.sqrt(propagator.dt / propagator.df) # the sqrt(dt/df) is due to the discrete nature of things 
                ASE_noise = ifft_(noise['noise_vector'] / np.sqrt(2) * ASE_expected_amplitude, propagator.dt)
                total_noise = total_noise + ASE_noise
            elif noise['noise_type'] == 'FWHM linewidth':
                # unlike the other noise options, this one changes the signal rather than adds to the total noise
                signal = signal * noise['noise_vector']
            else:
                raise ValueError(f"Noise type {noise['noise_type']} invalid")

        return signal + total_noise
    
    @property
    def propagator(self):
        return self._propagator
    
    @propagator.setter
    def propagator(self, propagator):
        self._propagator = propagator
        self.resample_noise(seed=self._seed)
    
    def resample_noise(self, seed=None):
        if (not AdditiveNoise.simulate_with_noise): # do not bother
            return

        if (self._propagator is None):
            raise ValueError("no propagator has been set")

        if (seed is not None):
            np.random.seed(seed)

        for noise_source in self.noise_sources:
            noise = np.random.normal(scale=1, size=(self.propagator.n_samples, 2)).view(dtype='complex')
            if noise_source['filter'] is not None:
                noise = noise_source['filter'].get_filtered_time(noise, self.propagator)
            
            if (noise_source['noise_type'] == 'absolute power'):
                noise_scaling = np.sqrt(noise_source['noise_param'] / np.mean(power_(noise))) # set average power to equal noise_param
                noise = noise * noise_scaling
            elif (noise_source['noise_type'] == 'rms constant'):
                noise_scaling = noise_source['noise_param'] / np.sqrt(np.mean(power_(noise)))
                noise = noise * noise_scaling
            elif (noise_source['noise_type'] == 'FWHM linewidth'):
                expected_phase_noise_amplitude = np.sqrt((noise_source['noise_param'] / np.pi) / 2) / ifft_shift_(np.abs(self.propagator.f))
                phase_noise = expected_phase_noise_amplitude * AdditiveNoise._get_real_noise_signal_freq(self.propagator)
                noise = ifft_(phase_noise, self.propagator.dt)
            noise_source['noise_vector'] = noise
            noise_source['mean_noise_power'] = np.mean(power_(noise)) # TODO: remind self what this is for...
    
    @staticmethod
    def get_OSNR(signal, noise, in_dB=True):
        """
        Returns OSNR (RMS(input)^2 / RMS(noise)^2)
        """
        osnr = np.mean(power_(signal)) / np.mean(power_(noise))
        if (not in_dB):
            return osnr

        return 10 * np.log10(osnr)
    
    @staticmethod
    def _get_real_noise_signal_freq(propagator):
        single_side_noise = np.random.normal(scale=1, size=(propagator.n_samples // 2, 2)).view(dtype='complex')
        double_side_noise = np.concatenate((single_side_noise, np.flip(np.conj(single_side_noise))))
        return double_side_noise

# ------------------------------------------------- Visualisation methods, to help with debugging ----------------------------------------
    def display_noisy_signal(self, signal, propagator=None):
        """
        Displays plots the noisy signal power in the time and frequency domains

        Note that signal WILL set sample_num value, but the vector signal WILL NOT be modified
        """
        #TODO: fix up the axes a bit but tbh I just want to see the shape
        _, ax = plt.subplots(2, 1)

        signal = np.copy(signal) # for ease of testing, we don't want to have to furnish multiple input signal objects

        noisy_signal = self.add_noise_to_propagation(signal, propagator)
        if (propagator is None): # use arbitrary scale
            raise ValueError('propagator cannot be None')

        ax[0].plot(propagator.t, power_(noisy_signal), label='time domain')
        ax[1].plot(propagator.f, 10 * np.log10(psd_(noisy_signal, propagator.dt, propagator.df)), label='freq domain')
        ax[0].legend()
        ax[1].legend()
        plt.title('Noisy signal')
        plt.show()
    
    def display_noise(self, signal, propagator=None):
        """
        Displays plots the noise power in the time and frequency domains

        Note that signal WILL set sample_num value, but the vector signal WILL NOT be modified
        """
        #TODO: fix up the axes a bit but tbh I just want to see the shape
        _, ax = plt.subplots(2, 1)

        if (propagator is None): # use arbitrary scale
            raise ValueError('propagator needs a value!')

        mutated_signal = np.copy(signal) # for ease of testing, we don't want to have to furnish multiple input signal objects
        noise_vector = self.add_noise_to_propagation(mutated_signal, propagator) - signal
    
        ax[0].plot(propagator.t, power_(noise_vector), label='time domain')
        ax[1].plot(propagator.f, 10 * np.log10(psd_(noise_vector, propagator.dt, propagator.df)), label='freq domain (log)')
        ax[0].legend()
        ax[1].legend()
        plt.title('Total noise')
        plt.show()

    def display_noise_sources_absolute(self, propagator=None):
        """
        Displays plots the power of individual noise contributions in the time and frequency domains

        Since no input vector is provided, outputs are displayed as if all noise source options were absolute power

        Raises exception if propagator is None
        """
        #TODO: fix up the axes a bit but tbh I just want to see the shape
    
        _, ax = plt.subplots(2, 1)

        if (self.propagator is None or self._propagator is not propagator):
            self.propagator = propagator
    
        for noise in self.noise_sources:
            noise_param = noise['noise_param']
            noise_vector = noise['noise_vector'] / noise['noise_param']**0.5 # will make it proportional to those relative vals
            ax[0].plot(propagator.t, power_(noise_vector), label=f'time domain, param: {noise_param}')
            ax[1].plot(propagator.f, psd_(noise_vector, propagator.dt, propagator.df), label=f'freq domain, param: {noise_param}')

        ax[0].legend()
        ax[1].legend()
        plt.title('Display each noise source as absolute')
        plt.show()