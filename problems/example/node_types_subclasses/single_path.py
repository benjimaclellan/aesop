"""

"""
import matplotlib.pyplot as plt

from pint import UnitRegistry
unit = UnitRegistry()
import autograd.numpy as np

from ..node_types import SinglePath

from ..assets.decorators import register_node_types_all
from ..assets.functions import fft_, ifft_, psd_, power_, fft_shift_

@register_node_types_all
class CorningFiber(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.node_lock = True

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

        self.beta = 1

        super().__init__(**kwargs)
        return

    # TODO : check this, and every other model for correctness (so far its been about logic flow)
    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):  # node propagate functions always take a list of propagators
        state = states[0]
        length = self.parameters[0]
        dispersion = fft_shift_(np.exp(-1j * length * self.beta * np.power(2 * np.pi * propagator.f, 2) ), ax=0)
        state = ifft_( dispersion * fft_(state, propagator.dt), propagator.dt)
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

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):  # node propagate functions always take a list of propagators
        state = states[0]

        depth = self.parameters[0]
        frequency = self.parameters[1]

        state1 = state * np.exp(1j * depth * (np.cos(2 * np.pi * frequency * propagator.t, dtype='complex')))
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

        #TODO: add test to make sure (at initialization that all these variables are the same length)
        # Then: also add one at runtime that ensure the .parameters variable is the same length
        self.number_of_parameters = 2 * number_of_bins

        self.default_parameters = [1] * number_of_bins + [0] * number_of_bins

        self.upper_bounds = [1] * number_of_bins + [2*np.pi] * number_of_bins
        self.lower_bounds = [0] * number_of_bins + [0] * number_of_bins
        self.data_types = 2 * number_of_bins * ['float']
        self.step_sizes = [None] * number_of_bins + [None] * number_of_bins
        self.parameter_imprecisions = [1] * number_of_bins + [2*np.pi] * number_of_bins
        self.parameter_units = [None] * number_of_bins + [unit.rad] * number_of_bins
        self.parameter_locks = 2 * self.number_of_parameters * [False]
        self.parameter_names = ['amplitude{}'.format(ind) for ind in range(number_of_bins)] + \
                               ['phase{}'.format(ind) for ind in range(number_of_bins)]


        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0):  # node propagate functions always take a list of propagators
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
        pad_left = np.zeros((left, 1))
        pad_right = np.zeros((N - right, 1))

        # Concatenate the arrays together
        # We cannot use array assignment as it is not supported by autograd
        amplitude_mask = np.concatenate((pad_left, amp1, pad_right), axis=0)
        phase_mask = np.concatenate((pad_left, phase1, pad_right), axis=0)
        mask = fft_shift_(amplitude_mask * np.exp(1j * phase_mask), ax=0)

        state = ifft_(mask * fft_(state, propagator.dt), propagator.dt)

        return [state]

@register_node_types_all
class DelayLine(SinglePath):
    """
    """
    def __init__(self, **kwargs):
        self.node_lock = False

        self._lengths = [1e-12, 2e-12, 4e-12, 8e-12, 16e-12, 32e-12, 128e-12, 64e-12]
        self._n = 1.444

        self.number_of_parameters = 8
        self.upper_bounds = [1] * self.number_of_parameters
        self.lower_bounds = [0] * self.number_of_parameters
        self.data_types = ['float'] * self.number_of_parameters
        self.step_sizes = [None] * self.number_of_parameters
        self.parameter_imprecisions = [0.01] * self.number_of_parameters
        self.parameter_units = [None] * self.number_of_parameters
        self.parameter_locks = [False] * self.number_of_parameters
        self.parameter_names = ['coupling_ratio{}'.format(ind) for ind in range(self.number_of_parameters)]

        self.default_parameters = [0] * self.number_of_parameters

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0):  # node propagate functions always take a list of propagators
        state = states[0]

        coupling_ratios = self.parameters

        # field is in the spectral domain here now
        field_short = fft_(state, propagator.dt)
        field_long = np.zeros_like(field_short)

        field_short_tmp = np.zeros_like(field_short)

        for i, coupling_ratio in enumerate(coupling_ratios):
            length = (propagator.speed_of_light / self._n) * self._lengths[i]
            beta = self._n * (2 * np.pi * (propagator.f + propagator.central_frequency)) / propagator.speed_of_light

            field_short_tmp = field_short
            try:
                field_short = (np.sqrt(1 - coupling_ratio) * field_short + 1j * np.sqrt(coupling_ratio) * field_long)
                field_long = np.exp(1j * beta * length) * (
                        1j * np.sqrt(coupling_ratio) * field_short_tmp + np.sqrt(1 - coupling_ratio) * field_long)
            except RuntimeWarning as w:
                print(f'RuntimeWarning: {w}')
                print(f'coupling ratios: {coupling_ratios}')
                print(f'coupling ratio: {coupling_ratio}')
                raise w
        return [ifft_(field_short, propagator.dt)]