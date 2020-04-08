#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from config.config import np

from problems.example.node_types import SinglePath

from ..assets.decorators import register_node_types_all


@register_node_types_all
class CorningFiber(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.number_of_parameters = 1
        self.upper_bounds = [10]
        self.lower_bounds = [0]
        self.data_types = ['float']
        self.step_size = [None]
        self.parameter_imprecision = [1]
        self.parameter_units = ['s']
        self.parameter_locked = [False]

        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):  # node propagate functions always take a list of propagators
        parameters = self.parameters
        state = states[0]
        return [state]

