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

from ..node_types import SourceModel

@register_node_types_all
class PulsedLaser(SourceModel):
    """
    """
    node_acronym = 'PL'
    number_of_parameters = 7

    def __init__(self, **kwargs):
        self.node_lock = False

        # self.default_parameters = ['gaussian', 3e-3, 1.0, 10e-9, 1.56e-6, True]
        self.default_parameters = ['gaussian', 100e-12, 1.0, 10e-9, 1.56e-6, True, 1e3]

        self.upper_bounds = [None, 1e-9, 1.0, 1/10e6, 1.56e-6, None, 5e9]
        self.lower_bounds = [None, 3e-11, 0.0001, 1/1e9, 1.54e-6, None, 0]
        self.data_types = ['str', 'float', 'float', 'float', 'float', 'bool', 'float']
        self.step_sizes = [None, None, None, None, None, None, None]
        self.parameter_imprecisions = [1, 10e-12, 0.1, 0.1e-9, 1e-10, 1, 1]
        self.parameter_units = [None, unit.s, unit.W, unit.s, unit.m, None, unit.Hz]
        self.parameter_locks = [True, False, False, False, True, True, True]
        self.parameter_names = ['pulse_shape', 'pulse_width', 'peak_power', 't_rep', 'central_wl', 'train', 'FWHM_linewidth']
        # self.parameter_symbols =[r"$x_{{"+f"{ind}"+r"}}$" for ind in range(self.number_of_parameters)]
        self.parameter_symbols =[r"$x$", r"$x_\tau$", r"$x_P$", r"$x_{trep}$", r"$x$", r"$x$", r"$x$", ]

        self.parameters = self.default_parameters
        super().__init__(**kwargs)
        self.set_parameters_as_attr()
        self.noise_model = AdditiveNoise(noise_param=self._FWHM_linewidth, noise_type='FWHM linewidth')
        return


    def get_pulse_train(self, t, pulse_width, rep_t, peak_power, pulse_shape='gaussian'):
        wrapped_t = np.sin(np.pi * t / rep_t)
        unwrapped_t = np.arcsin(wrapped_t) * rep_t / np.pi

        if pulse_shape == 'gaussian':
            pulse = self.gaussian(unwrapped_t, pulse_width)
        elif pulse_shape == 'sech':
            pulse = self.sech(unwrapped_t, pulse_width)
        else:
            raise RuntimeError(f"Pulsed Laser: {pulse_shape} is not a defined pulse shape")

        state = pulse * np.sqrt(peak_power)
        return state

    @staticmethod
    def sech(t, width):
        return 1 / np.cosh(t / width).astype('complex')

    @staticmethod
    def gaussian(t, width):
        return np.exp(-0.5 * (np.power(t / width, 2))).astype('complex')


    def propagate(self, state, propagator, save_transforms=False):
        self.set_parameters_as_attr()
        pulse_width = self._pulse_width / (2*np.sqrt(np.log(2)))  # check the scaling between envelope FWHM and power FWHM for Gaussian
        state = self.get_pulse_train(propagator.t, pulse_width, self._t_rep, self._peak_power, pulse_shape=self._pulse_shape)
        return state


@register_node_types_all
class ContinuousWaveLaser(SourceModel):
    """
    """
    node_acronym = 'CW'
    number_of_parameters = 4

    def __init__(self, **kwargs):
        self.node_lock = False

        self.default_parameters = [0.04, 1.55e-6, 55.0, 0.1e3] # default OSNR and linewidth from: https://www.nktphotonics.com/lasers-fibers/product/koheras-adjustik-low-noise-single-frequency-lasers/

        self.upper_bounds = [0.04, 1.56e-6, 200.0, 1.0e6] # upper bound for osnr randomly set
        self.lower_bounds = [1.0e-8, 1.54e-6, 1.0, 0.0] # note, autograd has a 'divide by zero warning' when CW lower bound set to 0.0 exactly. use slightly above to avoid. (also physically valid too)
        self.data_types = ['float', 'float', 'float', 'float']
        self.step_sizes = [None, None, None, None]
        self.parameter_imprecisions = [0.001e-3, 0.01e-6, 0, 0]
        self.parameter_units = [unit.W, unit.m, None, unit.Hz] # TODO: check whether we should use dB instead of None
        self.parameter_locks = [False, True, True, True]
        self.parameter_names = ['peak_power', 'central_wl', 'osnr_dB', 'FWHM_linewidth']

        self.parameter_symbols = [r"$x_P$", r"$x_\lambda$", r"$x_{SNR}$", r"$x_{FWHM}$"]
        self.parameters = self.default_parameters

        super().__init__(**kwargs)
        self.set_parameters_as_attr()
        self.update_noise_model()


    def propagate(self, state, propagator, save_transforms=False):
        peak_power = self.parameters[0]
        state = np.sqrt(peak_power) * np.ones_like(state)
        return state
    
    def update_noise_model(self):
        self.noise_model = AdditiveNoise(noise_type='FWHM linewidth', noise_param=self._FWHM_linewidth)
        self.noise_model.add_noise_source(noise_type='osnr', noise_param=self._osnr_dB)


# # @register_node_types_all
# class BitStream(SourceModel):
#     """
#     """
#
#     def __init__(self, seed=None, **kwargs):
#
#         random_state = np.random.RandomState(seed=seed)
#
#         self.node_lock = True
#         self.node_acronym = 'BIT'
#         self.number_of_parameters = 7
#         self.default_parameters = [10, 1.0, 12e-9, 100e-12, 1.55e-6, 55.0, 0.1e3]  #
#
#         self.upper_bounds = [50, 1.0, 12e-9, 100e-12, 1.56e-6, 200.0, 1.0e6]  #
#         self.lower_bounds = [2, 0.0, 4e-9, 10e-12, 1.54e-6, 1.0, 0.0]  #
#         self.data_types = ['int', 'float', 'float', 'float', 'float', 'float', 'float']
#         self.step_sizes = [1, None, None, None, None, None, None]
#         self.parameter_imprecisions = [0.0, 0.01e-6, 0, 0, 0, 0, 0]
#         self.parameter_units = [None, unit.W, unit.s, unit.m, None, unit.Hz]  # TODO: check whether we should use dB instead of None
#         self.parameter_locks = [True, True, True, True, True, True, True]
#         self.parameter_names = ['n_bits', 'peak_power', 't_rep', 'pulse_width', 'central_wl', 'osnr_dB', 'FWHM_linewidth']
#
#         self.parameter_symbols = [r"$x_"+f"{i}" + r"$" for i in range(self.number_of_parameters)]
#         self.parameters = self.default_parameters
#
#         super().__init__(**kwargs)
#         self.set_parameters_as_attr()
#         self.update_noise_model()
#
#         n_bits = self.parameters[0]
#         self.bits = random_state.randint(0, 2, n_bits)
#         self.pattern = None
#
#     def make_binary_pulse_pattern(self, propagator, bits):
#         n_bits = len(bits)
#         pattern = np.zeros_like(propagator.t)
#         for i, bit in enumerate(bits):
#             pattern += bit * np.sqrt(self._peak_power) * np.exp(-np.power((propagator.t - (i - (n_bits-1)/2) * self._t_rep)/self._pulse_width, 2))
#         return pattern
#
#     def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):
#         assert self.pattern.shape == states[0].shape
#         state = self.pattern
#         return [state]
#
#     def update_noise_model(self):
#         self.noise_model = AdditiveNoise(noise_type='FWHM linewidth', noise_param=self._FWHM_linewidth)
#         self.noise_model.add_noise_source(noise_type='osnr', noise_param=self._osnr_dB)