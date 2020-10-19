"""

"""

import matplotlib.pyplot as plt
import scipy.signal as sig
import autograd.numpy as np
from pint import UnitRegistry
unit = UnitRegistry()

from ..assets.decorators import register_node_types_all
from ..assets.functions import power_, psd_, fft_, ifft_, ifft_shift_
from ..assets.additive_noise import AdditiveNoise

from ..node_types import Input

@register_node_types_all
class PulsedLaser(Input):
    """
    """
    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'PL'
        self.number_of_parameters = 7

        # self.default_parameters = ['gaussian', 3e-3, 1.0, 10e-9, 1.56e-6, True]
        self.default_parameters = ['gaussian', 100e-12, 1.0, 10e-9, 1.56e-6, True, 1e3]

        self.upper_bounds = [None, 1e-9, 1.0, 1/10e6, 1.56e-6, None, 5e9]
        self.lower_bounds = [None, 3e-11, 0.0001, 1/1e9, 1.54e-6, None, 0]
        self.data_types = ['str', 'float', 'float', 'float', 'float', 'bool', 'float']
        self.step_sizes = [None, None, None, None, None, None, None]
        self.parameter_imprecisions = [1, 10e-12, 0.1, 0.1e-9, 0.01e-6, 1, 1]
        self.parameter_units = [None, unit.s, unit.W, unit.s, unit.m, None, unit.Hz]
        self.parameter_locks = [True, False, False, False, True, True, True]
        self.parameter_names = ['pulse_shape', 'pulse_width', 'peak_power', 't_rep', 'central_wl', 'train', 'FWHM_linewidth']
        self.parameter_symbols =[r"$x_{{"+f"{ind}"+r"}}$" for ind in range(self.number_of_parameters)]

        self.parameters = self.default_parameters
        super().__init__(**kwargs)
        self.set_parameters_as_attr()
        self.noise_model = AdditiveNoise(noise_param=self._FWHM_linewidth, noise_type='FWHM linewidth')
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):
        self.set_parameters_as_attr()
        width = self._pulse_width / (2*np.sqrt(np.log(2)))  # check the scaling between envelope FWHM and power FWHM for Gaussian

        def sech(t, width):
            return 1 / np.cosh(t / width).astype('complex')

        def gaussian(t, width):
            return np.exp(-0.5 * (np.power(t / width, 2))).astype('complex')

        # this is a way to allow the differentiation wrt repetition rate.
        # many Python and numpy functions give faulty derivatives (mod, impulses, setting array values)
        t = propagator.t
        wrapped_t = np.sin(np.pi * t / self._t_rep)
        unwrapped_t = np.arcsin(wrapped_t) * self._t_rep / np.pi

        if self._pulse_shape == 'gaussian':
            pulse = gaussian(unwrapped_t, width)
        elif self._pulse_shape == 'sech':
            pulse = sech(unwrapped_t, width)
        else:
            raise RuntimeError("Pulsed Laser: Not a defined pulse shape")

        state = pulse * np.sqrt(self._peak_power)
        return [state]



@register_node_types_all
class ContinuousWaveLaser(Input):
    """
    """

    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'CW'
        self.number_of_parameters = 4
        self.default_parameters = [0.04, 1.55e-6, 55.0, 0.1e3] # default OSNR and linewidth from: https://www.nktphotonics.com/lasers-fibers/product/koheras-adjustik-low-noise-single-frequency-lasers/

        self.upper_bounds = [0.04, 1.56e-6, 200.0, 1.0e6] # upper bound for osnr randomly set
        self.lower_bounds = [1.0e-8, 1.54e-6, 1.0, 0.0] # note, autograd has a 'divide by zero warning' when CW lower bound set to 0.0 exactly. use slightly above to avoid. (also physically valid too)
        self.data_types = ['float', 'float', 'float', 'float']
        self.step_sizes = [None, None, None, None]
        self.parameter_imprecisions = [0.1, 0.01e-6, 0, 0]
        self.parameter_units = [unit.W, unit.m, None, unit.Hz] # TODO: check whether we should use dB instead of None
        self.parameter_locks = [False, True, True, True]
        self.parameter_names = ['peak_power', 'central_wl', 'osnr_dB', 'FWHM_linewidth']

        self.parameter_symbols = [r"$x_P$", r"$x_\lambda$", r"$x_{SNR}$", r"$x_{FWHM}$"]
        self.parameters = self.default_parameters

        super().__init__(**kwargs)
        self.set_parameters_as_attr()
        self.update_noise_model()


    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):
        peak_power = self.parameters[0]
        state = np.sqrt(peak_power) * np.ones_like(states[0])
        return [state]
    
    def update_noise_model(self):
        self.noise_model = AdditiveNoise(noise_type='FWHM linewidth', noise_param=self._FWHM_linewidth)
        self.noise_model.add_noise_source(noise_type='osnr', noise_param=self._osnr_dB)