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
        self.default_parameters = [0.5]

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):
        coupling_ratio = self.parameters[0]
        if (num_inputs == 1) and (num_outputs == 2):
            state = states[0]
            return [(coupling_ratio) * state, (1-coupling_ratio) * state * np.exp(1j * np.pi/2)]

        elif (num_inputs == 2) and (num_outputs == 1):
            return [states[0] + states[1] * np.exp(1j * np.pi / 2)]

        else:
            raise ValueError("Not implemented yet: splitters should only be 2x1 or 1x2 for simplicity")