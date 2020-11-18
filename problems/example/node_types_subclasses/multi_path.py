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
    number_of_parameters = 1
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

        # self.upper_bounds = [1.0]
        # self.lower_bounds = [0.0]
        # self.data_types = ['float']
        # self.step_sizes = [None]
        # self.parameter_imprecisions = [0.1]
        # self.parameter_units = [None]
        # self.parameter_locks = [False]
        # self.parameter_names = ['coupling_ratio']
        # self.default_parameters = [0.5]
        # self.parameter_symbols = [r"$x_r$"]
        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        i, j = np.arange(0, num_inputs, 1), np.arange(0, num_outputs, 1)
        I, J = np.meshgrid(i, j)
        S = np.exp(1j * 2 * np.pi * I * J / num_outputs)
        states_tmp = np.stack(states, 1)

        states_scattered = np.matmul(S, states_tmp)
        states_scattered_lst = [states_scattered[:,i,:] for i in range(states_scattered.shape[1])]
        return states_scattered_lst

        # theta = self.parameters[0] * np.pi/2
        #
        # C = np.array([[np.cos(theta), -1j * np.sin(theta)],
        #               [-1j * np.sin(theta), np.cos(theta)]])
        #
        # if (num_inputs == 1) and (num_outputs == 2):
        #     state = states[0]
        #     return [C[0,0] * state, C[0,1] * state]
        # elif (num_inputs == 2) and (num_outputs == 1):
        #     return [C[1,0]*states[0] + C[0,0]*states[1]]
        # elif (num_inputs == 1) and (num_outputs == 1):
        #     return states
        # else:
        #     raise ValueError("Not implemented yet: splitters should only be 2x1 or 1x2 for simplicity")


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
