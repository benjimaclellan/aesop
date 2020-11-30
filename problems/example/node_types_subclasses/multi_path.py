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
    node_acronym = 'BS'
    number_of_parameters = 0
    def __init__(self, **kwargs):
        self.node_lock = False

        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_sizes = []
        self.parameter_imprecisions = []
        self.parameter_units = []
        self.parameter_locks = []
        self.parameter_names = []
        self.default_parameters = []
        self.parameter_symbols = []

    def update_attributes(self, num_inputs, num_outputs):
        num_parameters = num_outputs - 1

        self.upper_bounds = [1.0] * num_parameters
        self.lower_bounds = [0.0] * num_parameters
        self.data_types = ['float'] * num_parameters
        self.step_sizes = [None] * num_parameters
        self.parameter_imprecisions = [0.05] * num_parameters
        self.parameter_units = [None] * num_parameters
        self.parameter_locks = [False] * num_parameters
        self.parameter_names = [f'ratio-{i}' for i in range(num_parameters)]
        self.default_parameters = [1 - 1 / i for i in range(num_parameters+1, 1, -1)]
        self.parameter_symbols = [f'x_{i}' for i in range(num_parameters)]

        self.parameters = self.default_parameters
        return

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        DEBUG = False

        a = self.parameters
        w = [(1 - an) * np.product(a[:n]) for n, an in enumerate(a)] + [np.product(a)]
        if DEBUG: print(a, w, sum(w))
        i, j = np.arange(0, num_inputs, 1), np.arange(0, num_outputs, 1)
        I, J = np.meshgrid(i, j)
        I = I / num_inputs
        J = J / num_outputs

        _, X = np.meshgrid(i, np.sqrt(w))
        if DEBUG: print(X)

        S = X * np.exp(1j * np.pi * (I + J))
        if DEBUG: print(S)
        states_tmp = np.stack(states, 1)

        states_scattered = np.matmul(S, states_tmp)
        states_scattered_lst = [states_scattered[:,i,:] for i in range(states_scattered.shape[1])]
        return states_scattered_lst


# @register_node_types_all
# class WavelengthDivisionMultiplexer(MultiPath):
#     node_lock = False
#     node_acronym = 'WDM'
#     def __init__(self, **kwargs):
#
#
#         self.number_of_parameters = 1
#         self.upper_bounds = [1.0]
#         # self.upper_bounds = [196.27e12]
#         # self.lower_bounds = [191.25e12]
#         self.lower_bounds = [-1.0]
#         self.data_types = ['float']
#         self.step_sizes = [None]
#         self.parameter_imprecisions = [1]
#         self.parameter_units = [None]
#         self.parameter_locks = [False]
#         self.parameter_names = ['splitting_frequency']
#         self.default_parameters = [0.0]
#         self.parameter_symbols = [r"$x_f$"]
#
#         super().__init__(**kwargs)
#         return
#
#     def propagate(self, states, propagator, num_inputs = 1, num_outputs = 2, save_transforms=False):
#         frequency_split = self.parameters[0] # propagator.f[0] + (self.parameters[0]) * (propagator.f[-1] - propagator.f[0])
#         k = 500
#         logistic = 1.0 / (1.0 + np.exp(-k * (propagator.f / propagator.f[-1] - frequency_split)))
#
#         if save_transforms:
#             self.transform = (('f', logistic, 'wdm'), )
#         else:
#             self.transform = None
#
#         if (num_inputs == 1) and (num_outputs == 2):
#             state = states[0]
#             left_path = ifft_(ifft_shift_(logistic) * fft_(state, propagator.dt), propagator.dt)
#             right_path = ifft_(ifft_shift_(1.0-logistic) * fft_(state, propagator.dt), propagator.dt)
#             return [left_path, right_path]
#         elif (num_inputs == 2) and (num_outputs == 1):
#             return [states[0] + states[1]]
#         elif (num_inputs == 1) and (num_outputs == 1):
#             return states
#         else:
#             raise ValueError("Not implemented yet: splitters should only be 2x1 or 1x2 for simplicity")
#
