"""

"""

import matplotlib.pyplot as plt

from config.config import np

from pint import UnitRegistry
unit = UnitRegistry()

from ..assets.decorators import register_node_types_all
from ..assets.functions import power_, psd_

from problems.example.node_types import Input


def sech(t, offset, width):
    return 1/np.cosh((t + offset) / width, dtype='complex')

def gaussian(t, offset, width):
    return np.exp(-0.5 * (np.power((t + offset) / width, 2)), dtype='complex')

@register_node_types_all
class PulsedLaser(Input):
    """
    """

    def __init__(self, **kwargs):
        self.number_of_parameters = 5
        self.default_parameters = ['gaussian', 10e-9, 1, 1e-8, 1.55e-6]
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_size = []
        self.parameter_imprecision = []
        self.parameter_units = [None, unit.s, unit.W, unit.s, unit.m]
        self.parameter_locked = [True, True, True, True, True]
        self.parameter_names = ['pulse_shape', 'pulse_width', 'peak_power', 't_rep', 'central_wl']

        self.parameters = self.default_parameters

        super().__init__(**kwargs)
        return



    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):
        self.set_parameters_as_attr()

        n_pulses = int(np.ceil(propagator.window_t / self._t_rep))
        train = True

        # create initial train of Gaussian pulses
        if self._pulse_shape == 'gaussian':
            pulse_function = gaussian
        elif self._pulse_shape == 'sech':
            pulse_function = sech
        else:
            raise AttributeError("This is not a valid pulse shape")

        state = pulse_function(propagator.t, 0, self._pulse_width)
        if train:
            for i_pulse in list(range(-1, -(n_pulses // 2 + 1), -1)) + list( range(1, n_pulses // 2 + 1, +1)):  # fill in all pulses except central one
                state += pulse_function(propagator.t, propagator.window_t * (i_pulse / n_pulses), self._pulse_width)

        # scale by peak_power power
        state *= np.sqrt(self._peak_power)

        return [state]



@register_node_types_all
class ContinuousWaveLaser(Input):
    """
    """

    def __init__(self, **kwargs):
        self.number_of_parameters = 2
        self.default_parameters = [1, 1.55e-6]
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_size = []
        self.parameter_imprecision = []
        self.parameter_units = [unit.W, unit.m]
        self.parameter_locked = [True, True]
        self.parameter_names = ['peak_power', 'central_wl']

        self.parameters = self.default_parameters

        super().__init__(**kwargs)
        return



    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):
        self.set_parameters_as_attr()
        state = np.sqrt(self._peak_power) * np.ones_like(states[0])
        return [state]