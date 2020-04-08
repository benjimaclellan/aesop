#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import autograd.numpy as np

from ..assets.decorators import register_node_types_all
from problems.example.node_types import Input


def sech(x):
    return 1/np.cosh(x, dtype='complex')


@register_node_types_all
class PulsedLaser(Input):
    """
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.number_of_parameters = 0
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_size = []
        self.parameter_imprecision = []
        self.parameter_units = []
        self.parameter_locked = []


        # if propagator is None: raise ValueError("Pulse class requires a Propagator class as an input")

        # self._pulse_shape = pulse_shape
        # self._pulse_width = pulse_width
        # self._peak_power = peak_power
        # self._t_rep = t_rep
        # self._train = train
        # self._n_pulses = int(np.ceil(propagator.window_t / t_rep))
        return

    #TODO: How do we want to design this nodetype?

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):

        state = states[0]

        # propagator.state = np.zeros_like(propagator.state)
        #
        # # create initial train of Gaussian pulses
        # if self._pulse_shape == 'gaussian':
        #     propagator.state = np.exp(-0.5 * (np.power(propagator.t / self._pulse_width, 2)))
        #     if self._train:
        #         for i_pulse in list(range(-1, -(self._n_pulses // 2 + 1), -1)) + list(
        #                 range(1, self.n_pulses // 2 + 1, +1)):  # fill in all pulses except central one
        #             propagator.state += np.exp(-0.5 * (
        #                 np.power((propagator.t + (propagator.window_t * (i_pulse / self._n_pulses))) / self._pulse_width, 2)))
        #
        # # create initial train of sech2 pulses
        # elif self._pulse_shape == 'sech':
        #     propagator.state = sech(propagator.t / self._pulse_width)
        #     if self.train:
        #         for i_pulse in list(range(-1, -self._n_pulses // 2 + 1, -1)) + list(
        #                 range(1, self._n_pulses // 2 - 1, +1)):  # fill in all pulses except central one
        #             propagator.state += sech((propagator.t + (propagator.window_t * (i_pulse / self._n_pulses))) / self._pulse_width)
        #
        # else:
        #     raise ValueError("Please specify a valid pulse shape")
        #
        # # scale by peak_power power
        # propagator.state *= np.sqrt(self._peak_power)

        return [state]
