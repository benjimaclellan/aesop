#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from config.config import np

from problems.example.node_types import MultiPath

from ..assets.decorators import register_node_types_all


@register_node_types_all
class PhaseModulator(MultiPath):
    def __init__(self, **kwargs):
        self.number_of_parameters = 2
        self.upper_bounds = [10, 10]
        self.lower_bounds = [0, 0]
        self.data_types = ['float', 'float']
        self.step_size = [None, None]
        self.parameter_imprecision = [1, 1]
        self.parameter_units = ['', '']
        self.parameter_locked = [False, False]
        self.parameter_names = ['mod_depth', 'mod_freq']

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):
        parameters = self.parameters
        state = states[0]
        return [state]
