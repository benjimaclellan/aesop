"""

"""
import matplotlib.pyplot as plt

from pint import UnitRegistry
unit = UnitRegistry()
import autograd.numpy as np

from ..node_types import SinglePath

from ..assets.decorators import register_node_types_all
from ..assets.functions import fft_, ifft_, psd_, power_, ifft_shift_

@register_node_types_all
class CorningFiber(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.node_lock = False

        self.number_of_parameters = 1
        self.default_parameters = [1]

        self.upper_bounds = [1000]
        self.lower_bounds = [0]
        self.data_types = ['float']
        self.step_sizes = [None]
        self.parameter_imprecisions = [1]
        self.parameter_units = [unit.m]
        self.parameter_locks = [False]
        self.parameter_names = ['length']

        self.beta = 1e-23

        super().__init__(**kwargs)
        return

    # TODO : check this, and every other model for correctness (so far its been about logic flow)
    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]
        length = self.parameters[0]
        dispersion = length * self.beta * np.power(2 * np.pi * propagator.f, 2)

        if save_transforms:
            self.transform = (('f', dispersion, 'dispersion'),)
        else:
            self.transform = None

        state = ifft_( ifft_shift_(np.exp(-1j * dispersion), ax=0) * fft_(state, propagator.dt), propagator.dt)
        return [state]



@register_node_types_all
class PhaseModulator(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.node_lock = False

        self.number_of_parameters = 2
        self.default_parameters = [1, 12e9]

        self.upper_bounds = [20, 12e9]
        self.lower_bounds = [0, 1e9]
        self.data_types = ['float', 'float']
        self.step_sizes = [None, 1e9]
        self.parameter_imprecisions = [1, 1]
        self.parameter_units = [unit.rad, unit.Hz]
        self.parameter_locks = [False, True]
        self.parameter_names = ['depth', 'frequency']

        super().__init__(**kwargs)

        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]

        depth = self.parameters[0]
        frequency = self.parameters[1]

        transform = depth * (np.cos(2 * np.pi * frequency * propagator.t))
    
        if save_transforms:
            self.transform = (('t', transform, 'modulation'),)
        else:
            self.transform = None

        state1 = state * np.exp(1j * transform)
        return [state1]


@register_node_types_all
class WaveShaper(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.node_lock = False

        number_of_bins = 5
        self._number_of_bins = number_of_bins
        self.frequency_bin_width = 12e9
        self.extinction_ratio = 10 **( -35 / 10)

        #TODO: add test to make sure (at initialization that all these variables are the same length)
        # Then: also add one at runtime that ensure the .parameters variable is the same length
        self.number_of_parameters = 2 * number_of_bins

        self.default_parameters = [1] * number_of_bins + [0] * number_of_bins

        self.upper_bounds = [1] * number_of_bins + [2*np.pi] * number_of_bins
        self.lower_bounds = [self.extinction_ratio] * number_of_bins + [0] * number_of_bins
        self.data_types = 2 * number_of_bins * ['float']
        self.step_sizes = [None] * number_of_bins + [None] * number_of_bins
        self.parameter_imprecisions = [1] * number_of_bins + [2*np.pi] * number_of_bins
        self.parameter_units = [None] * number_of_bins + [unit.rad] * number_of_bins
        self.parameter_locks = 2 * self.number_of_parameters * [False]
        self.parameter_names = ['amplitude{}'.format(ind) for ind in range(number_of_bins)] + \
                               ['phase{}'.format(ind) for ind in range(number_of_bins)]

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]

        # Slice at into the first half (amp) and last half (phase)
        amplitudes = self.parameters[:self._number_of_bins]
        phases = self.parameters[self._number_of_bins:]

        n = np.floor(propagator.n_samples / ((1 / propagator.dt) / self.frequency_bin_width)).astype('int')
        N = np.shape(propagator.f)[0]
        tmp = np.ones((n, 1))

        a = np.array([i * tmp for i in amplitudes])
        p = np.array([i * tmp for i in phases])

        amp1 = np.concatenate(a)
        phase1 = np.concatenate(p)
    
        left = np.floor((propagator.n_samples - amp1.shape[0]) / 2).astype('int')
        right = propagator.n_samples - np.ceil((propagator.n_samples - amp1.shape[0]) / 2).astype('int')

        # we will pad amp1 and phase1 with zeros so they are the correct size
        pad_left = np.ones((left, 1)) * self.extinction_ratio
        pad_right = np.ones((N - right, 1)) * self.extinction_ratio

        # Concatenate the arrays together
        # We cannot use array assignment as it is not supported by autograd
        amplitude_mask = np.concatenate((pad_left, amp1, pad_right), axis=0)
        phase_mask = np.concatenate((pad_left, phase1, pad_right), axis=0)

        state = ifft_(ifft_shift_(amplitude_mask * np.exp(1j * phase_mask), ax=0) * fft_(state, propagator.dt), propagator.dt)

        if save_transforms:
            self.transform = (('f', amplitude_mask, 'amplitude'), ('f', phase_mask, 'phase'),)
        else:
            self.transform = None

        return [state]


@register_node_types_all
class DelayLine(SinglePath):
    """
    """
    def __init__(self, **kwargs):
        self.node_lock = False

        self.number_of_parameters = 8
        self.upper_bounds = [1] * self.number_of_parameters
        self.lower_bounds = [0] * self.number_of_parameters
        self.data_types = ['float'] * self.number_of_parameters
        self.step_sizes = [None] * self.number_of_parameters
        self.parameter_imprecisions = [0.01] * self.number_of_parameters
        self.parameter_units = [None] * self.number_of_parameters
        self.parameter_locks = [False] * self.number_of_parameters
        self.parameter_names = ['coupling_ratio{}'.format(ind) for ind in range(self.number_of_parameters)]

        self._n = 1.444
        self._delays = [2**k * 1e-12 for k in range(0, self.number_of_parameters)]
        # [1e-12, 2e-12, 4e-12, 8e-12, 16e-12, 32e-12, 128e-12, 64e-12

        self.default_parameters = [0] * self.number_of_parameters

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]
        coupling_ratios = self.parameters

        # field is in the spectral domain here now
        field_short = fft_(state, propagator.dt)
        field_long = np.zeros_like(field_short)

        field_short_tmp = np.zeros_like(field_short)

        ## TODO fix the fft, ifft functions
        for (coupling_ratio, delay) in zip(coupling_ratios, self._delays):
            length = (propagator.speed_of_light / self._n) * delay
            beta = self._n * (2 * np.pi * (propagator.f + propagator.central_frequency)) / propagator.speed_of_light

            field_short_tmp = field_short

            try:
                field_short = (np.sqrt(1 - coupling_ratio) * field_short + 1j * np.sqrt(coupling_ratio) * field_long)
                field_long = np.exp(1j * ifft_shift_(beta * length, ax=0)) * (
                        1j * np.sqrt(coupling_ratio) * field_short_tmp + np.sqrt(1 - coupling_ratio) * field_long)
            except RuntimeWarning as w:
                print(f'RuntimeWarning: {w}')
                print(f'coupling ratios: {coupling_ratios}')
                print(f'coupling ratio: {coupling_ratio}')
                raise w

        if save_transforms:
            transform = np.zeros_like(propagator.t).astype('float')
            for (coupling_ratio, delay) in zip(coupling_ratios, self._delays):
                transform[(np.abs(propagator.t - (-delay))).argmin()] = coupling_ratio
            self.transform = (('t', transform, 'delay coupling ratios'),)
        else:
            self.transform = None

        return [ifft_(field_short, propagator.dt)]


# @register_node_types_all
class EDFA(SinglePath):
    """
    EDFA modelled as follows:

    ---------------g(f), the small-signal gain---------------
    
    In the following equations, f = peak frequency - actual frequency
    g(f) = g_max * exp(-f^2/Beta)
    With Beta = d^2/(ln(10^(G_f / 10))),
    d = max distance between peak wavelength and the edge of the band,
    Gf = gain flatness in dB,

    Such that, as expected, g(d) = g_min

    --------------- G(f), the true gain---------------
    Defined according to this model:
    https://www.researchgate.net/publication/3289918_Simple_black_box_model_for_erbium-doped_fiber_amplifiers [1]

    TODO: figure out how to handle too powerful input signals (which are 'invalid inputs')
    """
    def __init__(self, max_small_signal=30, peak_wl=1550e-9, band=(1530e-9, 1565e-9), P_out_max=20,
                 gain_flatness=1.5, alpha=1):
        """

        :param max_small_signal: maximum small signal gain, in dB (assumed to be at band center)
        :param peak_wl: peak wavelength of the EDFA where max amplification occurs, in m (max amplification there)
        :param band: (min wavelength, max_wavelength), in m
        :param P_out_max: maximum output power, in dBm
        :param gain_flatness : gain flatness, in dB (Gmax(dB) - Gmin(dB))
        :param alpha : parameter alpha, as defined in [1] (default value of 1 seems reasonable)
        """
        pass

    def propagate(self, states, propagator, num_inputs=1, num_outputs=1, save_transforms=False):  # node propagate functions always take a list of propagators
        """
        """
        # TODO: deal with the transform shtick
        state = states[0]
        pass
        return state
    
    def _gain(self, state):
        """
        """
        # P_in = average state power
        # return self.small_signal / (1 + (self.small_signal * P_in / self.P_max)**self.alpha)
        return 0
    
    @property
    def _small_signal_gain(self):
        pass
        return 0