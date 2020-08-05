import autograd.numpy as np
import matplotlib.pyplot as plt

from .functions import fft_

"""
TODO: <to add as time moves on>
TODO: consider phase?
TODO: consider my choice to make all central / transition wavelengths RELATIVE to central carrier
"""

class AmplitudeFilter():
    """
    AmplitudeFilter class: supports both static methods and an object which can save presets
    Applies amplitude filter to input vectors. Note that the filter is always normalized to one (for now)

    Does not change the phase of signals
    
    Resources:
    https://www.edmundoptics.com/knowledge-center/application-notes/optics/optical-filters/
    """
    def __init__(self, shape='exponential of square', **shape_params):
        """
        Initializes amplitude only filter with a given parametrised shape.

        :param shape: 'exponential of square', 'gaussian', 'longpass', 'bandpass', 'shortpass'
        :param **shape_params: any parameters required for the shape
            exponential of square: f(x) = exp(-f^2/beta), shape_params = central_wl_rel, FWHM
            gaussian: shape_params = central_wl_rel (i.e. c / mean_freq), std_dev_wl
            longpass: f(x) = 1 / (1 + e^(-A (x - a))), shape_params = slope (# of Hz from 0.1 to 0.9 transmission), transition_wl_rel
            bandpass: longpass + shortpass, shape_params = slope, transition_wl_small, transition_wl_large
            shortpass: f(x) = 1 / (1 + e^(A (x - a))), shape_params = slope (# of Hz from 0.1 to 0.9 transmission), transition_wl_rel

            TODO: add lorentzian, poissonian
        """
        self.shape = shape
        self.shape_params = shape_params

    def get_filtered_freq(self, state_rf, propagator):
        """
        Takes in frequency vector, and returns filtered frequency vector

        :param state_rf: frequency vector
        :return : the frequency vector subject to the filter
        """
        pass

    def get_filtered_time(self, state, propagator):
        pass

    @property
    def sample_num(self):
        if (self._sample_num is None):
            raise ValueError('No sample num set yet')
        return self._sample_num
    
    @sample_num.setter
    def sample_num(self, n):
        self._sample_num = n
        self._generate_filter()
    
    def _generate_filter(self, propagator):
        if self.shape == 'exponential of square':
            self.filter = AmplitudeFilter._exp_of_square(propagator, **self.shape_params)
        elif self.shape == 'gaussian':
            self.filter = AmplitudeFilter._gaussian(propagator, **self.shape_params)
        elif self.shape == 'longpass':
            self.filter = AmplitudeFilter._longpass(propagator, **self.shape_params)
        elif self.shape == 'shortpass':
            self.filter = AmplitudeFilter._shortpass(propagator, **self.shape_params)
        elif self.shape == 'bandpass':
            self.filter = AmplitudeFilter._bandpass(propagator, **self.shape_params)
        else:
            raise ValueError(f'{self.shape} filter type not implemented!')
    
    @staticmethod
    def get_filtered_freq_static(state_rf, shape='exponential of square', **shape_params):
        pass

    @staticmethod
    def get_filtered_time_static(state_rf, shape='exponential of square', **shape_params):
        pass

    @staticmethod
    def _exp_of_square(n, central_wl=1550e-9, FWHM=60e-9):
        beta = -1 * np.power(FWHM, 2) / 4 / np.log(0.5)
        

    @staticmethod
    def _gaussian(n,central_wl=1550e-9, std_dev_wl=60e-9):
        pass

    @staticmethod
    def _longpass(n, transition_wl=1550e-9, slope=60e-9):
        pass 

    @staticmethod
    def _shortpass(n, transition_wl=1550e-9, slope=60e-9):
        pass

    @staticmethod
    def _bandpass(n, transition_wl_rel_small=-20e-9, transition_wl_rel_large=20e-9, slope=15e-9):
        return AmplitudeFilter._longpass(n, transition_wl=transition_wl_small, slope=slope) * \
               AmplitudeFilter._shortpass(n, transition_wl=transition_wl_large, slope=slope)
    


