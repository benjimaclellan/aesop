"""

"""

from pint import UnitRegistry
unit = UnitRegistry()
import autograd.numpy as np
import matplotlib.pyplot as plt

from ..node_types import MultiPath

from ..assets.decorators import register_node_types_all
from ..assets.functions import power_, psd_, fft_, ifft_, ifft_shift_

@register_node_types_all
class VariablePowerSplitter(MultiPath):
    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'BS'

        self.number_of_parameters = 1
        self.upper_bounds = [0.9]
        self.lower_bounds = [0.1]
        self.data_types = ['float']
        self.step_sizes = [None]
        self.parameter_imprecisions = [1]
        self.parameter_units = [None]
        self.parameter_locks = [False]
        self.parameter_names = ['coupling_ratio']
        self.default_parameters = [0.5]
        self.parameter_symbols = [r"$x_r$"]
        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 2, save_transforms=False):
        coupling_ratio = self.parameters[0]
        if (num_inputs == 1) and (num_outputs == 2):
            state = states[0]
            return [(coupling_ratio) * state, (1-coupling_ratio) * state * np.exp(1j * np.pi/2)]
        elif (num_inputs == 2) and (num_outputs == 1):
            return [states[0] + states[1] * np.exp(1j * np.pi / 2)]
        elif (num_inputs == 1) and (num_outputs == 1):
            return states
        else:
            raise ValueError("Not implemented yet: splitters should only be 2x1 or 1x2 for simplicity")



@register_node_types_all
class WavelengthDivisionMultiplexer(MultiPath):
    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'WDM'

        self.number_of_parameters = 1
        self.upper_bounds = [1.0]
        # self.upper_bounds = [196.27e12]
        # self.lower_bounds = [191.25e12]
        self.lower_bounds = [-1.0]
        self.data_types = ['float']
        self.step_sizes = [None]
        self.parameter_imprecisions = [1]
        self.parameter_units = [None]
        self.parameter_locks = [False]
        self.parameter_names = ['splitting_frequency']
        self.default_parameters = [0.0]
        self.parameter_symbols = [r"$x_f$"]

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 2, save_transforms=False):
        frequency_split = self.parameters[0] # propagator.f[0] + (self.parameters[0]) * (propagator.f[-1] - propagator.f[0])
        k = 500
        logistic = 1.0 / (1.0 + np.exp(-k * (propagator.f / propagator.f[-1] - frequency_split)))

        if save_transforms:
            self.transform = (('f', logistic, 'wdm'), )
        else:
            self.transform = None

        if (num_inputs == 1) and (num_outputs == 2):
            state = states[0]
            left_path = ifft_(ifft_shift_(logistic) * fft_(state, propagator.dt), propagator.dt)
            right_path = ifft_(ifft_shift_(1.0-logistic) * fft_(state, propagator.dt), propagator.dt)
            return [left_path, right_path]
        elif (num_inputs == 2) and (num_outputs == 1):
            return [states[0] + states[1]]
        elif (num_inputs == 1) and (num_outputs == 1):
            return states
        else:
            raise ValueError("Not implemented yet: splitters should only be 2x1 or 1x2 for simplicity")

