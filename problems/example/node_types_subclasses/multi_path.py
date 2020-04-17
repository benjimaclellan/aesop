"""

"""

from pint import UnitRegistry
unit = UnitRegistry()
import autograd.numpy as np

from ..node_types import MultiPath

from ..assets.decorators import register_node_types_all


@register_node_types_all
class VariablePowerSplitter(MultiPath):
    def __init__(self, **kwargs):
        self.node_lock = False

        self.number_of_parameters = 1
        self.upper_bounds = [1]
        self.lower_bounds = [0]
        self.data_types = ['float']
        self.step_sizes = [None]
        self.parameter_imprecisions = [1]
        self.parameter_units = [None]
        self.parameter_locks = [False]
        self.parameter_names = ['coupling_ratio']
        self._default_parameters = [0.5]

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0):
        self.set_parameters_as_attr()
        print(self._mod_depth)
        state = sum(states)
        return [state]
