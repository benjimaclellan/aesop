#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

# import autograd.numpy as np
from config.config import np

from problems.example.node_types import Output


class MeasurementDevice(Output):
    def __init__(self):
        super().__init__()

        self.number_of_parameters = 0
        self.parameters = []
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_size = []
        self.parameter_imprecision = []
        self.parameter_units = []
        self.parameter_locked = []

        return

    def propagate(self, propagators):  # node propagate functions always take a list of propagators
        propagator = propagators[0]
        parameters = self.parameters
        propagator.state = propagator.state
        return [propagator]

