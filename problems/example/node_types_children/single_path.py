"""

"""
import matplotlib.pyplot as plt

from pint import UnitRegistry
unit = UnitRegistry()

from config.config import np

from problems.example.node_types import SinglePath

from ..assets.decorators import register_node_types_all
from ..assets.functions import fft_, ifft_, psd_, power_

@register_node_types_all
class CorningFiber(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.number_of_parameters = 1
        self.upper_bounds = [10]
        self.lower_bounds = [0]
        self.data_types = ['float']
        self.step_size = [None]
        self.parameter_imprecision = [1]
        self.parameter_units = [unit.m]
        self.parameter_locked = [False]
        self.parameter_names = ['length']
        self._default_parameters = [1]

        self.beta = 1

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):  # node propagate functions always take a list of propagators
        state = states[0]
        length = self.parameters[0]
        state = ifft_( np.exp(-1j * length * self.beta * np.power(2 * np.pi * propagator.f) ) * fft_(state, propagator.dt), propagator.dt)
        return [state]



@register_node_types_all
class PhaseModulator(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.number_of_parameters = 2
        self.upper_bounds = [10, 12e9]
        self.lower_bounds = [0, 1e9]
        self.data_types = ['float', 'float']
        self.step_size = [None, 1e9]
        self.parameter_imprecision = [1, 1]
        self.parameter_units = [unit.rad, unit.Hz]
        self.parameter_locked = [False, False]
        self.parameter_names = ['depth', 'frequency']
        self._default_parameters = [1, 12e9]

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):  # node propagate functions always take a list of propagators
        state = states[0]

        depth = self.parameters[0]
        frequency = self.parameters[1]

        state *= np.exp(1j * depth * (np.cos(2 * np.pi * frequency * propagator.t, dtype='complex')))
        return [state]