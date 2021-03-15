import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.constants import elementary_charge
from scipy import interpolate
from scipy.optimize import curve_fit

from .functions import fft_, ifft_, power_, psd_, ifft_shift_

"""
TODO: can we create different copies of the class for threadsafe-ness purposes? Think about potential parallelization (one class per graph?)
      Also consider modifying the graph class to contain the "simulate with noise" variable, such that it can be determined on a per-graph basis
      rather than for all AdditiveNoise objects at once
"""

def lorentzian(xdata, HWHM):
    return HWHM / np.pi / (np.power(xdata, 2) + np.power(HWHM, 2))


def inverse_x_squared(xdata, c):
    return c / np.power(xdata, 2)


class AdditiveNoise():
    """
    Additive noise is noise modelled as: <total signal>(t) = <pure signal>(t) + <noise>(t)

    This class allows for the shape of noise to be defined, and added to signal states
    """

    simulate_with_noise = True
    supported_noise_dist = ['gaussian']
    supported_noise_types = ['osnr', 'absolute power', 'edfa ASE', 'rms', 'shot', 'FWHM linewidth', 'phase noise from linewidth']

    def __init__(self, distribution='gaussian', noise_param=1, noise_filter=None, noise_type='osnr', noise_on=True, seed=None):
        """
        Creates an noise object with the described properties

        :param distribution: type of noise distribution. Only Gaussian is supported at this time
        :param noise_param: noise parameter (significance depends on noise type)
        :param noise_filter: filter to apply on AWGN to get desired spectral shape (describes only shape, not amplitude)
        :param noise_type: 'osnr' (noise_param = OSNR in dB), 'absolute power' (noise_param = average power in Watts),
                           'edfa ASE' (noise_param = edfa object), 'rms' (noise_param = rms value of signal),
                           'shot' (noise_param[0] = slope of noise (with respect to input power), noise_param[1] = intercept of noise)
                           'FWHM linewdith' (noise_param = FWHM linewidth)
                           'phase noise from linewidth' (noise_param = FWHM linewidth). This noise acts differently, in that it's not automatically included in propagation
        :param noise_on: True if the noise of the object is to be applied, False otherwise
                         Note that the noise can also be turned off for the whole class (upon which this variable has no effect)
        :param seed: seed with which np.random can generate the pseudorandom noise vectors

        :raises ValueError if (1) non-gaussian distribution is requested
                              (2) noise_type is not 'osnr', 'absolute power', 'edfa ASE', 'rms', 'shot', 'FWHM linewidth', 'phase noise from linewidth'
        """
        self._propagator = None  # defined the first time that noise is added to the propagation
        
        self._seed = seed # can't actually seed it now because we have no idea what random calls will come between initialization and the first call to add_noise_to_propagate
    
        self.noise_sources = [] # keep log of noise sources for debugging, and in case of resampling

        # create noise distribution
        self.add_noise_source(distribution=distribution, noise_param=noise_param, noise_filter=noise_filter, noise_type=noise_type)

        self.noise_on = noise_on
    
    def add_noise_source(self, distribution='gaussian', noise_param=1, noise_filter=None, noise_type='osnr'):
        """
        Adds a noise source to the AdditiveNoise object (all future calls to add_noise_to_propagation will also include this source)
        WARNING: all noise sources should probably be added at once (and non-concurrently please, this is not threadsafe yet) so that seeding actually makes results predictable
                 We will seed ONCE when initializing the noise sources, so if you have an external random call, might make it harder to track down the precise noise vectors 

        :param distribution: type of noise distribution. Only Gaussian is supported at this time
        :param noise_param: noise parameter (significance depends on noise type)
        :param noise_filter: filter to apply on AWGN to get desired spectral shape (describes only shape, not amplitude)
        :param noise_type: 'osnr' (noise_param = OSNR in dB), 'absolute power' (noise_param = average power in Watts),
                           'edfa ASE' (noise_param = edfa object), 'rms' (noise_param = rms value of signal),
                           'shot' (noise_param[0] = slope of noise (with respect to input power), noise_param[1] = intercept of noise)
                           'FWHM linewdith' (noise_param = FWHM linewidth)
                           'phase noise from linewidth' (noise_param = FWHM linewidth). This noise acts differently, in that it's not automatically included in propagation

        :raises ValueError if (1) non-gaussian distribution is requested (2) someone has the audacity of asking for a filter I have yet to implement
                              (3) noise_type is not 'osnr', 'absolute power', 'edfa ASE', 'rms', 'shot', 'FWHM linewidth'
        """
        if (distribution not in AdditiveNoise.supported_noise_dist):
            raise ValueError(f"{distribution} distribution not supported. Noise distribution must be in {AdditiveNoise.supported_noise_dist}")
        
        if (noise_type not in AdditiveNoise.supported_noise_types):
            raise ValueError(f'Noise type must be in {AdditiveNoise.supported_noise_types}. {noise_type} is not valid')

        if (self.propagator is not None):
            noise = self._get_noise_vector(noise_filter, noise_type, noise_param)
            mean_noise_power = np.mean(power_(noise))
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

        self.noise_sources.append(source)
    
    def add_noise_to_propagation(self, signal, propagator):
        """
        Adds noise to the input state, and returns the noisy state. Every noise source that has been added to the object is added to the signal
        
        :param signal: the pre-noise state
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

            elif noise['noise_type'] == 'absolute power' or noise['noise_type'] == 'rms':
                total_noise = total_noise + noise['noise_vector']

            elif noise['noise_type'] == 'shot':
                # v_rms^2 = 2qBR^2(I_d + I_p), I_rms^2 = 2qB(I_d + I_p)
                # set bandwidth to full available frequency if there is no filter
                bandwidth = np.sum(np.abs(noise['filter'].filter)) * propagator.df if noise['filter'] is not None else propagator.n_samples * propagator.df
                target_squared_rms = (noise['noise_param'][0] * np.mean(np.abs(signal)) + noise['noise_param'][1]) * bandwidth
                scaling_factor = np.sqrt(target_squared_rms / np.mean(power_(noise['noise_vector'])))
                total_noise = total_noise + scaling_factor * noise['noise_vector']

            elif noise['noise_type'] == 'edfa ASE':
                ASE_expected_amplitude = noise['noise_param'].get_ASE_filter(propagator) * np.sqrt(propagator.dt / propagator.df) # the sqrt(dt/df) is due to the discrete nature of things 
                ASE_noise = ifft_(noise['noise_vector'] / np.sqrt(2) * ASE_expected_amplitude, propagator.dt)
                total_noise = total_noise + ASE_noise

            elif noise['noise_type'] == 'FWHM linewidth':
                signal = signal * noise['noise_vector'] # unlike the other noise options, this one changes the signal rather than adds to the total noise
            
            elif noise['noise_type'] == 'phase noise from linewidth':
                pass # phase noise can be extracted separately, is not meant to be summed into propagation
            else:
                raise ValueError(f"Noise type {noise['noise_type']} invalid")

        return signal + total_noise
    
    def get_phase_noise(self, propagator):
        """
        Returns a time-domain distribution with average value centre_freq, such that a linewidth as described by the
        FWHM linewidth noise type is applied. Assumes only one linewidth noise type in the object, and that the linewidth has a Lorentzian shape

        :param centre_freq: centre frequency
        :param propagator: propagator (length must be matched)

        :returns the frequency with some variation due to linewidth
        :raises ValueError if no FWHM linewidth noise source has been added to this AdditiveNoise object
        """
        if not AdditiveNoise.simulate_with_noise:
            return np.zeros(propagator.n_samples).reshape(propagator.n_samples, 1)

        if (self.propagator is None or propagator is not self.propagator): # we avoid resampling the noise when the propagator hasn't changed, for efficiency
            self.propagator = propagator

        for noise in self.noise_sources:
            if noise['noise_type'] == 'phase noise from linewidth':  
                return noise['noise_vector']
        
        raise ValueError('No FWHM linewidth noise type in this AdditiveNoise object!')
    
    @property
    def propagator(self):
        return self._propagator
    
    @propagator.setter
    def propagator(self, propagator):
        self._propagator = propagator
        self.resample_noise(seed=self._seed)
    
    def resample_noise(self, seed=None):
        """
        Resamples all noise sources pseudo-randomly. The new noise will be sampled from the same underlying distribution as the old

        :param seed: seed to set np.random's pseudo-random number generation
        """
        if (not AdditiveNoise.simulate_with_noise): # do not bother
            return

        if (self._propagator is None):
            raise ValueError("no propagator has been set")

        if (seed is not None):
            np.random.seed(seed)

        for noise_source in self.noise_sources:
            noise_source['noise_vector'] = self._get_noise_vector(noise_source['filter'], noise_source['noise_type'], noise_source['noise_param'])
            noise_source['mean_noise_power'] = np.mean(power_(noise_source['noise_vector']))
    
    def _get_noise_vector(self, noise_filter, noise_type, noise_param):
        """
        Retrieve a noise vector with appropriate scaling (if scaling is independent of input vector: otherwise scaling occurs at a later stage)

        :param noise_filter: the filter to apply to noise prior to scaling
        :noise_type: as described in the constructor docstring
        :noise_param: as described in the constructor docstring

        :return: a correctly shaped (and potentially correctly scaled) vector
        """
        noise = np.random.normal(scale=1, size=(self.propagator.n_samples, 2)).view(dtype='complex')
        if noise_filter is not None:
            noise = noise_filter.get_filtered_time(noise, self.propagator)

        if (noise_type == 'absolute power'):
            noise_scaling = np.sqrt(noise_param / np.mean(power_(noise))) # set average power to equal noise_param
            noise = noise * noise_scaling
        elif (noise_type == 'rms'):
            noise_scaling = noise_param / np.sqrt(np.mean(power_(noise)))
            noise = noise * noise_scaling
        elif (noise_type == 'FWHM linewidth'):
            if (noise_param == 0): # mathematically the same without the if-else, but it stops autograd errors
                noise = np.ones(self.propagator.n_samples).reshape(self.propagator.n_samples, 1)
            else:
                expected_phase_noise_amplitude = np.sqrt((noise_param / np.pi) / 2) / ifft_shift_(np.abs(self.propagator.f)) * np.sqrt(self.propagator.dt)
                phase_noise = expected_phase_noise_amplitude * AdditiveNoise._get_real_noise_signal_freq(self.propagator)
                noise = np.exp(1j * np.real(ifft_(phase_noise, self.propagator.dt)))
        elif (noise_type == 'phase noise from linewidth'):
            if (noise_param == 0):
                noise = np.zeros(self.propagator.n_samples).reshape(self.propagator.n_samples, 1)
            else:
                expected_phase_noise_amplitude = np.sqrt((noise_param / np.pi) / 2) / ifft_shift_(np.abs(self.propagator.f)) * np.sqrt(self.propagator.dt)
                phase_noise = expected_phase_noise_amplitude * AdditiveNoise._get_real_noise_signal_freq(self.propagator) 
                noise = np.real(ifft_(phase_noise, self.propagator.dt))

        return noise
    
    @staticmethod
    def get_OSNR(signal, noise, in_dB=True):
        """
        Returns OSNR (RMS(signal)^2 / RMS(noise)^2)

        :param signal: the pure signal
        :param noise: the noise
        :in_dB: if True, OSNR is returned in dB. If False, it is returned as the ratio above
        """
        osnr = np.mean(power_(signal)) / np.mean(power_(noise))
        if (not in_dB):
            return osnr

        return 10 * np.log10(osnr)
    
    @staticmethod
    def _get_real_noise_signal_freq(propagator):
        """
        Get a frequency domain signal, normalized to average power 1. In the time domain, this signal is real

        :param propagator: the propagator which the length of our output must match
        """
        time_noise = np.random.normal(scale=1, size=(propagator.n_samples, 1))
        freq_noise = np.fft.fft(time_noise, axis=0)
        return freq_noise

    @staticmethod
    def get_estimated_FWHM_linewidth_from_points(phase_psd_points):
        """
        Returns an estimate of the FWHM linewidth, from fitting phase psd to c / f^2. This results in a Lorentzian lineshape

        :param phase_psd_points: phase psd points to fit to, in dBc/Hz
        :return estimated FWHM (assumption is that linewidth is Lorentzian)
        """
        # _fit_phase_psd_from_points returns the best coefficient c such that the phase matches to c/f^2
        # c = h0/2, assuming uniform frequency spectrum (see noise simulation document)
        # with the same assumption, FWHM linewidth is pi * h0 = 2 * pi * c
        return AdditiveNoise._fit_phase_psd_from_points(phase_psd_points)[0] * 2 * np.pi

    @staticmethod
    def _fit_phase_psd_from_points(phase_psd_points, return_psd=False):
        """
        Returns an array of the phase PSD given some psd points.
        Fits a Lorentzian function for linewidth (i.e. phase PSD is proportional to c / f^2, frequency psd is constant)

        :param phase_psd_points: list of tuples (offset in Hz, phase noise in dBc/Hz)
        :param return_psd: function returns phase PSD only if True
        :return: fit_coefficient (the c in c/f^2), psd (if return_psd = True). Note that fit coefficient is returned as array, in case we want different fitting functions later
        """
        offsets = np.array([point[0] for point in phase_psd_points])
        offsets = np.concatenate((-1 * np.flip(offsets), offsets))
        dBc_per_Hz = np.array([point[1] for point in phase_psd_points]) # this is L(f) in usual units
        dBc_per_Hz = np.concatenate((np.flip(dBc_per_Hz), dBc_per_Hz))
        phase_psd = AdditiveNoise.dBc_per_Hz_to_rad2_per_Hz(dBc_per_Hz)

        fit_coeff, _ = curve_fit(inverse_x_squared, offsets, phase_psd)
        
        if return_psd:
            return fit_coeff, phase_psd
        return fit_coeff
    
    @staticmethod
    def dBc_per_Hz_to_rad2_per_Hz(dBc_per_Hz):
        return 2 * np.power(10, dBc_per_Hz / 10)


# ------------------------------------------------- Visualisation methods, to help with debugging ----------------------------------------
    def display_noisy_signal(self, signal, propagator):
        """
        Displays plots the noisy signal power in the time and frequency domains

        Note that signal WILL set sample_num value, but the vector signal WILL NOT be modified
        """
        _, ax = plt.subplots(2, 1)

        noisy_signal = self.add_noise_to_propagation(signal, propagator)

        ax[0].plot(propagator.t, ifft_shift_(power_(noisy_signal)), label='time domain')
        psd = psd_(noisy_signal, propagator.dt, propagator.df)
        psd_log = 10 * np.log10(psd / np.max(psd))
        ax[1].plot(propagator.f, psd_log, label='freq domain')
        ax[0].legend()
        ax[1].legend()
        plt.title('Noisy signal')
        plt.show()
    
    def display_noise(self, signal, propagator):
        """
        Displays plots the noise power in the time and frequency domains

        Note that signal WILL set sample_num value, but the vector signal WILL NOT be modified
        """
        _, ax = plt.subplots(2, 1)

        noise_vector = self.add_noise_to_propagation(signal, propagator) - signal
    
        ax[0].plot(propagator.t, ifft_shift_(power_(noise_vector)), label='time domain')
        psd = psd_(noise_vector, propagator.dt, propagator.df)
        psd_log =  10 * np.log10(psd / np.max(psd))
        ax[1].plot(propagator.f, psd_log, label='freq domain (log)')
        ax[0].legend()
        ax[1].legend()
        plt.title('Total noise')
        plt.show()

    def display_noise_sources_absolute(self, propagator):
        """
        Displays plots the power of individual noise contributions in the time and frequency domains

        Since no input vector is provided, outputs are displayed as if all noise source options were absolute power

        Raises exception if propagator is None
        """    
        _, ax = plt.subplots(2, 1)

        if (self.propagator is None):
            self.propagator = propagator

        normalization = max([np.max(psd_(noise['noise_vector'], propagator.dt, propagator.df)) for noise in self.noise_sources])

        for noise in self.noise_sources:
            noise_param = noise['noise_param']
            noise_vector = noise['noise_vector'] 
            ax[0].plot(propagator.t, power_(noise_vector), label=f'time domain, param: {noise_param}')
            ax[1].plot(propagator.f, 10 * np.log10(psd_(noise_vector, propagator.dt, propagator.df) / normalization), label=f'freq domain, param: {noise_param}')

        ax[0].legend()
        ax[1].legend()
        plt.title('Display each noise source as absolute')
        plt.show()