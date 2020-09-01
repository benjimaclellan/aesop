"""

"""

import matplotlib.pyplot as plt
import autograd.numpy as np
from pint import UnitRegistry
unit = UnitRegistry()

from ..assets.decorators import register_node_types_all
from ..assets.functions import power_, psd_, ifft_, ifft_shift_
from ..assets.additive_noise import AdditiveNoise

from ..node_types import Input


def sech(t, offset, width):
    return 1/np.cosh((t + offset) / width, dtype='complex')

def gaussian(t, offset, width):
    return np.exp(-0.5 * (np.power((t + offset) / width, 2)), dtype='complex')

@register_node_types_all
class PulsedLaser(Input):
    """
    """

    def __init__(self, **kwargs):
        self.node_lock = True

        self.number_of_parameters = 6

        self.upper_bounds = [None] * self.number_of_parameters
        self.lower_bounds = [None] * self.number_of_parameters
        self.data_types = [None] * self.number_of_parameters
        self.step_sizes = [None] * self.number_of_parameters
        self.parameter_imprecisions = [None] * self.number_of_parameters
        self.parameter_units = [None, unit.s, unit.W, unit.s, unit.m, None]
        self.parameter_locked = [True, True, True, True, True, True]
        self.parameter_names = ['pulse_shape', 'pulse_width', 'peak_power', 't_rep', 'central_wl', 'train']

        self.default_parameters = ['gaussian', 10e-9, 1, 1e-7, 1.55e-6, True]

        self.parameters = self.default_parameters

        super().__init__(**kwargs)
        return



    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):
        self.set_parameters_as_attr()

        n_pulses = int(np.ceil(propagator.window_t / self._t_rep))

        width = self._pulse_width /np.sqrt(2)*np.log(2)  # check the scaling between envelope FWHM and power FWHM for Gaussian

        # create initial train of Gaussian pulses
        if self._pulse_shape == 'gaussian':
            pulse_function = gaussian
        elif self._pulse_shape == 'sech':
            pulse_function = sech
        else:
            raise AttributeError("This is not a valid pulse shape")

        state = pulse_function(propagator.t, 0, width)
        if self._train:
            for i_pulse in list(range(-1, -(n_pulses // 2 + 1), -1)) + list( range(1, n_pulses // 2 + 1, +1)):  # fill in all pulses except central one
                state += pulse_function(propagator.t, propagator.window_t * (i_pulse / n_pulses), width)

        # scale by peak_power power
        state *= np.sqrt(self._peak_power)

        return [state]



@register_node_types_all
class ContinuousWaveLaser(Input):
    """
    """

    def __init__(self, **kwargs):
        self.node_lock = False

        self.number_of_parameters = 3
        self.default_parameters = [1, 1.55e-6, 55, 0.1e3] # default OSNR and linewidth from: https://www.nktphotonics.com/lasers-fibers/product/koheras-adjustik-low-noise-single-frequency-lasers/

        self.upper_bounds = [2, 1.54e-6, 200, 1e6] # upper bound for osnr randomly set
        self.lower_bounds = [0, 1.56e-6, 1, 0]
        self.data_types = ['float', 'float', 'float', 'float']
        self.step_sizes = [None, None, None, None]
        self.parameter_imprecisions = [0.1, 0.01e-6, 0, 0]
        self.parameter_units = [unit.W, unit.m, None, unit.Hz] # TODO: check whether we should use dB instead of None
        self.parameter_locks = [True, True, True, True]
        self.parameter_names = ['peak_power', 'central_wl', 'osnr_dB', 'FWHM_linewidth']

        self.parameters = self.default_parameters

        super().__init__(**kwargs)
        self.set_parameters_as_attr()

        # TODO: integrate both together
        self.noise_model = AdditiveNoise(noise_type='FWHM linewidth', noise_param=self._FWHM_linewidth)
        self.noise_model.add_noise_source(noise_param=self._osnr_dB)

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):
        state = np.sqrt(self._peak_power) * np.ones_like(states[0])
        return [state]
