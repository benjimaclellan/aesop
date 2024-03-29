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
    node_lock = False

    def __init__(self, **kwargs):

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
        super().__init__(**kwargs)

    def update_attributes(self, num_inputs, num_outputs):
        num_parameters = num_outputs - 1
        self.number_of_parameters = num_parameters

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
        if DEBUG: print(f'PowerSplitter', num_inputs, num_outputs, len(self.parameters), self.parameters)

        a = self.parameters
        w = np.array([(1 - an) * np.product(a[:n]) for n, an in enumerate(a)] + [np.product(a)])
        if DEBUG: print(a, w, sum(w))

        i, j = np.arange(0, num_inputs, 1), np.arange(0, num_outputs, 1)
        I, J = np.meshgrid(i, j)
        I = I / num_inputs
        J = J / num_outputs

        X = np.matmul(np.expand_dims(np.sqrt(np.array(w)), axis=1), np.expand_dims(np.ones_like(i), axis=0))
        if DEBUG: print(X)

        S = X / np.sqrt(num_outputs) * np.exp(-1j * 2 * np.pi * (I + J))
        if DEBUG: print(S)
        states_tmp = np.stack(states, 1)

        states_scattered = np.matmul(S, states_tmp)
        states_scattered_lst = [states_scattered[:, i, :] for i in range(states_scattered.shape[1])]
        return states_scattered_lst


# @register_node_types_all
class FrequencySplitter(MultiPath):
    node_acronym = 'FS'
    number_of_parameters = 0
    node_lock = False

    def __init__(self, **kwargs):
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
        super().__init__(**kwargs)

    def update_attributes(self, num_inputs, num_outputs):
        num_parameters = num_outputs - 1
        self.number_of_parameters = num_parameters

        self.upper_bounds = [1.0] * num_parameters
        self.lower_bounds = [0.0] * num_parameters
        self.data_types = ['float'] * num_parameters
        self.step_sizes = [None] * num_parameters
        self.parameter_imprecisions = [0.05] * num_parameters
        self.parameter_units = [None] * num_parameters
        self.parameter_locks = [False] * num_parameters
        self.parameter_names = [f'ratio-{i}' for i in range(num_parameters)]
        self.default_parameters = [1 - 1 / i for i in range(num_parameters + 1, 1, -1)]
        self.parameter_symbols = [f'x_{i}' for i in range(num_parameters)]

        self.parameters = self.default_parameters
        return

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        DEBUG = False
        if DEBUG: print(f'FrequencySplitter', num_inputs, num_outputs, len(self.parameters), self.parameters)

        state = np.sum(np.stack(states, 1), axis=1)

        a = self.parameters
        g = np.array([0] + [(1 - an) * np.product(a[:n]) for n, an in enumerate(a)] + [np.product(a)])
        w = np.array([sum(g[:n]) for n in range(1, len(g))] + [1])
        left_cutoffs, right_cutoffs = w[:-1], w[1:]

        k = 500
        new_states = []
        tmp_x = np.linspace(0, 1, propagator.f.shape[0]).reshape(propagator.f.shape)
        for j, (left_cutoff, right_cutoff) in enumerate(zip(left_cutoffs, right_cutoffs)):
            logistic = ((1.0 / (1.0 + np.exp(-k * (tmp_x - left_cutoff))))
                        * (1.0 / (1.0 + np.exp(k * (tmp_x - right_cutoff)))))
            new_states.append(ifft_(ifft_shift_(logistic) * fft_(state, propagator.dt), propagator.dt))
        return new_states
