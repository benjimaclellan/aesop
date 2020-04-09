"""

"""

import matplotlib.pyplot as plt

from config.config import np

from problems.example.node_types import Output

from ..assets.decorators import register_node_types_all


@register_node_types_all
class MeasurementDevice(Output):
    def __init__(self, **kwargs):
        self.number_of_parameters = 0
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_size = []
        self.parameter_imprecision = []
        self.parameter_units = []
        self.parameter_locked = []
        self.parameter_names = []
        self._default_parameters = []

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):  # node propagate functions always take a list of propagators
        return states
