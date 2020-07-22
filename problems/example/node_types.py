"""

"""

from lib.base_classes import NodeType as NodeTypeParent
from lib.decorators import register_node_types


@register_node_types
class Input(NodeTypeParent):
    """ Parent class for Input node types. These nodes have no incoming edges, and
    generally represent optical sources (pulsed laser, cw laser, etc)
    """

    __internal_var = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._node_type = "input node"

        self._range_input_edges = (0, 0)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (1, 4)  # minimum, maximum number of input edges, may be changed in children


@register_node_types
class Output(NodeTypeParent):
    """Parent class for Output node types. These nodes have no outgoing edges, and
    generally represent measurement devices
    """

    __internal_var = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._node_type = "output node"

        self._range_input_edges = (1, 3)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (0, 0)  # minimum, maximum number of input edges, may be changed in children

        return


@register_node_types
class MultiPath(NodeTypeParent):
    """Parent class for MultiPath node types. These nodes can have more than one incoming and/or outgoing
     edges - for example beamsplitters, discretized gratings, etc
    """

    __internal_var = 4

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

    __internal_var = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._node_type = "single-path node"

        self._range_input_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children

        return



