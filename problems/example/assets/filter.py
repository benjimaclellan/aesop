import autograd.numpy as np
import matplotlib.pyplot as plt

from .functions import fft_

"""
TODO: <to add as time moves on>
"""

class Filter():
    """
    Filter class: supports both static methods and an object which can save presets
    Applies filter to input vectors. Note that the filter is always normalized to one (for now)
    
    Resources:
    https://www.edmundoptics.com/knowledge-center/application-notes/optics/optical-filters/
    """
    def __init__(self, shape='exponential of square', **shape_params):
        """
        Initializes filter with a given parametrised shape

        :param shape: 'exponential of square', 'gaussian', 'longpass', 'bandpass', 'shortpass'
        :param **shape_params: any parameters required for the shape
            exponential of square: f(x) = exp(-f^2/beta), shape_params = central_wl, FWHM
            gaussian: shape_params = central_wl (i.e. c / mean_freq), std_dev (Hz)
            longpass: f(x) = 1 / (1 + e^(-A (x - a))), shape_params = slope (# of Hz from 0.1 to 0.9 transmission), transition_wl
            bandpass: longpass + shortpass, shape_params = central_wavelength, FWHM
            shortpass: f(x) = 1 / (1 + e^(A (x - a))), shape_params = slope (# of Hz from 0.1 to 0.9 transmission), transition_wl

            TODO: add lorentzian, poissonian
        """
        pass

    def get_filtered_freq(self, state_rf):
        pass

    def get_filtered_time(self, state):
        pass
    
    @staticmethod
    def get_filtered_freq_static(state_rf, shape='exponential of square', **shape_params):
        pass

    @staticmethod
    def get_filtered_time_static(shape, shape='exponential of square', **shape_params):
        pass


