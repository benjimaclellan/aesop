#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from config.config import np
# import autograd.numpy as np

from problems.example.node_types import SinglePath


class CorningFiber(SinglePath):
    """

    """

    def __init__(self):
        super().__init__()

        self.number_of_parameters = 1
        self.parameters = [10]
        self.upper_bounds = [10]
        self.lower_bounds = [0]
        self.data_types = ['float']
        self.step_size = [None]
        self.parameter_imprecision = [1]
        self.parameter_units = ['s']
        self.parameter_locked = [False]

        return

    def propagate(self, propagators):  # node propagate functions always take a list of propagators
        # if len(propagators) > 1:
        #     raise TypeError("Too many incoming propagators for this model")

        propagator = propagators[0]
        parameters = self.parameters
        propagator.state = propagator.state
        return [propagator]

