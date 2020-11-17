"""

"""
from lib.base_classes import NodeType as NodeTypeParent, TerminalNode
from lib.decorators import register_node_types

@register_node_types
class SourceModel(NodeTypeParent):
    """ Parent class for Input node types. These nodes have no incoming edges, and
    generally represent optical sources (pulsed laser, cw laser, etc)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._node_type = "input node"

        self._range_input_edges = (0, 0)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children


@register_node_types
class SinkModel(NodeTypeParent):
    """Parent class for Output node types. These nodes have no outgoing edges, and
    generally represent measurement devices
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._node_type = "output node"

        self._range_input_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (0, 0)  # minimum, maximum number of input edges, may be changed in children

        return


@register_node_types
class MultiPath(NodeTypeParent):
    """Parent class for MultiPath node types. These nodes can have more than one incoming and/or outgoing
     edges - for example beamsplitters, discretized gratings, etc
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._node_type = "multi-path node"

        self._range_input_edges = (1, 2)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (1, 2)  # minimum, maximum number of input edges, may be changed in children
        return




@register_node_types
class SinglePath(NodeTypeParent):
    """Parent class for SinglePath node types. These nodes have one incoming edge and one outgoing edge,
     and represent optical components such as fiber, phase-modulators, etc
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._node_type = "single-path node"

        self._range_input_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children

        return



# @register_node_types
class TerminalSource(NodeTypeParent):
    """
    """

    default_parameters = []
    number_of_parameters = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._node_type = "source node"

        self._range_input_edges = (0, 0)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children

        return

    def propagate(self, states, propagator, num_inputs=0, num_outputs=1, save_transforms=False):
        return states


# @register_node_types
class TerminalSink(NodeTypeParent):
    """
    """

    default_parameters = []
    number_of_parameters = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._node_type = "sink node"

        self._range_input_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (0, 0)  # minimum, maximum number of input edges, may be changed in children

        return

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):
        return states