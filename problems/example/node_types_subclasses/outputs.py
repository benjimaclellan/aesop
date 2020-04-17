"""

"""

import matplotlib.pyplot as plt
import autograd.numpy as np

from ..node_types import Output

from ..assets.decorators import register_node_types_all


@register_node_types_all
class MeasurementDevice(Output):
    def __init__(self, **kwargs):
        self.node_lock = True

        self.number_of_parameters = 0
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_sizes = []
        self.parameter_imprecisions = []
        self.parameter_units = []
        self.parameter_locks = []
        self.parameter_names = []

        self.default_parameters = []

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):
        return states
