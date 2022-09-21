from lib.base_classes import NodeType as NodeTypeParent
from simulator.fiber.assets.decorators import register_node_types_all, register_node_types_including_terminals


@register_node_types_including_terminals
class TerminalSource(NodeTypeParent):
    """
    """

    default_parameters = []
    number_of_parameters = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.node_acronym = 'SRC'
        self._node_type = "source node"

        self.node_lock = True

        self.default_parameters = []
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_sizes = []
        self.parameter_imprecisions = []
        self.parameter_units = []
        self.parameter_locks = []
        self.parameter_names = []

        self.parameter_symbols = []
        self.parameters = self.default_parameters

        self._range_input_edges = (0, 0)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children

        return

    def update_attributes(self, num_inputs, num_outputs):
        return

    def propagate(self, states, propagator, num_inputs=0, num_outputs=1, save_transforms=False):
        return states


@register_node_types_including_terminals
class TerminalSink(NodeTypeParent):
    """
    """

    default_parameters = []
    number_of_parameters = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.node_acronym = 'SNK'
        self._node_type = "sink node"

        self.node_lock = True

        self.default_parameters = []
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_sizes = []
        self.parameter_imprecisions = []
        self.parameter_units = []
        self.parameter_locks = []
        self.parameter_names = []

        self.parameter_symbols = []
        self.parameters = self.default_parameters

        self._range_input_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (0, 0)  # minimum, maximum number of input edges, may be changed in children

        return

    def update_attributes(self, num_inputs, num_outputs):
        return

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):
        return states