import autograd.numpy as np
import matplotlib.pyplot as plt

from .functions import fft_, ifft_, ifft_shift_

"""
TODO: compare the object based method execution speed (Averaged) against the static based functions. Decide which to keep accordingly
TODO: consider using the matlab bandpass rather than the sigmoids
"""

class Filter():
    """
    AmplitudeFilter class: supports both static methods and an object which can save presets
    Applies amplitude filter to input vectors. Note that the filter is always normalized to one (for now)

    Does not change the phase of signals
    
    Resources:
    https://www.edmundoptics.com/knowledge-center/application-notes/optics/optical-filters/
    """
    def __init__(self, shape='butterworth lowpass', **shape_params):
        """
        Initializes filter with a given parametrised shape.

        :param shape: 'exponential of square', 'gaussian', 'longpass', 'bandpass', 'shortpass', 'butterworth lowpass'
        :param **shape_params: any parameters required for the shape
            exponential of square: f(x) = exp(-f^2/beta), shape_params = central_wl_rel, FWHM
            gaussian: shape_params = central_wl_rel (i.e. c / mean_freq), std_dev_wl
            longpass: half of shape_params = slope (# of Hz from 0.1 to 0.9 transmission), transition_wl_rel
            bandpass: longpass + shortpass, shape_params = slope, transition_wl_small, transition_wl_large
            shortpass: f(x) = 1 / (1 + e^(A (x - a))), shape_params = slope (# of Hz from 0.1 to 0.9 transmission), transition_wl_rel

            TODO: add lorentzian, poissonian
        """
        self.shape = shape
        self.shape_params = shape_params
        self._propagator = None
        self._filter = None

    def get_filtered_freq(self, state_rf, propagator):
        """
        Takes in frequency vector, and returns filtered frequency vector

        :param state_rf: frequency vector
        :return : the frequency vector subject to the filter
        """
        if self.propagator is None or self.propagator is not propagator:
            self.propagator = propagator
        
        return state_rf * self.filter

    def get_filtered_time(self, state, propagator):
        state_rf = fft_(state, propagator.dt)
        output = self.get_filtered_freq(state_rf, propagator)
        return ifft_(output, propagator.dt)
    
    def display_filter(self, propagator):
        _, ax = plt.subplots(2, 1)
        
        if self.propagator is None or self.propagator is not propagator:
            self.propagator = propagator
    
        ax[0].plot(propagator.f, ifft_shift_(np.abs(self.filter)))
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Filter amplitude')

        ax[1].plot(propagator.f, ifft_shift_(np.angle(self.filter)))
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Filter phase')
        plt.title(f'Filter {self.shape}')
        plt.show()
    
    def display_filtered_output(self, state, propagator):
        filtered_time = self.get_filtered_time(state, propagator)
        filtered_freq = self.get_filtered_freq(fft_(state, propagator.dt), propagator)

        _, ax = plt.subplots(2, 1)
        ax[0].plot(propagator.t, state, label='unfiltered')
        ax[0].plot(propagator.t, filtered_time, label='filtered')
        ax[1].plot(propagator.f, fft_(state, propagator.dt), label='unfiltered')
        ax[1].plot(propagator.f, ifft_shift_(filtered_freq), label='filtered')
        ax[0].set_xlabel('time (s)')
        ax[1].set_ylabel('frequency (Hz)')
        plt.title(f'Filter {self.shape}')
        plt.show()

    @property
    def propagator(self):
        return self._propagator
    
    @propagator.setter
    def propagator(self, propagator):
        self._propagator = propagator
        self._filter = None # reset the filter, so that it remakes it according to the proper propagator
    
    @property
    def filter(self):
        if self._filter is None:
            self._filter = Filter.get_filter(self.propagator, self.shape, **self.shape_params)
        
        return self._filter
    
    @staticmethod
    def get_filtered_freq_static(state_rf, propagator, shape='exponential of square', **shape_params):
        return state_rf * Filter.get_filter(propagator, shape=shape, **shape_params)

    @staticmethod
    def get_filtered_time_static(state, propagator, shape='exponential of square', **shape_params):
        state_rf = fft_(state, propagator.dt)
        output = state_rf * Filter.get_filter(propagator, shape=shape, **shape_params)
        return ifft_(output, propagator.dt)

    @staticmethod
    def get_filter(propagator, shape='butterworth lowpass', **shape_params):
        if shape == 'exponential of square':
            return Filter._exp_of_square(propagator, **shape_params)
        elif shape == 'butterworth lowpass':
            return Filter._butterworth_lowpass(propagator, **shape_params)
        elif shape == 'gaussian':
            return Filter._gaussian(propagator, **shape_params)
        elif shape == 'longpass':
            return Filter._longpass(propagator, **shape_params)
        elif shape == 'shortpass':
            return Filter._shortpass(propagator, **shape_params)
        elif shape == 'bandpass':
            return Filter._bandpass(propagator, **shape_params)
        else:
            raise ValueError(f'{shape} filter type not implemented!')

    @staticmethod
    def _exp_of_square(propagator, central_wl=1550e-9, FWHM=50e-9):
        delta_f = propagator.speed_of_light * ( 1 / (central_wl - FWHM / 2) - 1 / (central_wl + FWHM / 2))

        beta = np.power(delta_f, 2) / 4 / np.log(2)
        central_freq = propagator.speed_of_light / central_wl
        f = ifft_shift_(propagator.f + propagator.central_frequency - central_freq)

        return np.exp(-1 * np.power(f, 2) / beta)

    @staticmethod
    def _gaussian(propagator, central_wl=1550e-9, std_dev_f=60e-9):
        raise ValueError('gaussian not yet implemented')

    @staticmethod
    def _longpass(propagator, transition_wl=1550e-9, slope=60e-9):
        raise ValueError('longpass not yet implemented')

    @staticmethod
    def _shortpass(propagator, transition_wl=1550e-9, slope=60e-9):
        raise ValueError('shortpass not yet implemented')

    @staticmethod
    def _bandpass(propagator, transition_wl_small=-20e-9, transition_wl_large=20e-9, slope=15e-9):
        raise ValueError('bandpass not yet implemented')
    
    @staticmethod
    def _butterworth_lowpass(propagator, transition_f=1e9, dc_gain=1, order=2):
        return ifft_shift_(dc_gain / Filter._butterworth_polynomial(1j * propagator.f / transition_f, order))
    
    @staticmethod
    def _butterworth_polynomial(s, n):
        if n < 0:
            raise ValueError(f'Order of the polynomial >= 0, cannot be {n}')
        
        product = 1
        max_num = n // 2
        if n % 2 != 0:
            product = s + 1
            max_num = (n - 1) // 2
        
        for k in range(1, max_num + 1):
            product = product * (s**2 - 2 * s * np.cos((2 * k + n - 1) / (2 * n) * np.pi) + 1)
        
        return product





