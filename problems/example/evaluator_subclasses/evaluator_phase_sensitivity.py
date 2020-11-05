""" """

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_
from lib.functions import scale_units

class PhaseSensitivity(Evaluator):
    """  """

    def __init__(self, propagator, phase=0.0, phase_node='ps', **kwargs):
        super().__init__(**kwargs)
        self.phase_node = phase_node
        self.phase = phase
        self.target = np.ones_like(propagator.t)


    def evaluate_graph(self, graph, propagator):
        measurement_node = -1 #graph.get_output_node()  # finds node with no outgoing edges

        def _function(_phase, _graph, _propagator, _node, _measurement_node):
            _graph.nodes[_node]['model'].parameters = [_phase]
            _graph.propagate(_propagator)
            _state = _graph.measure_propagator(_measurement_node)
            p = np.sum(power_(_state))
            return p
        _f = lambda x: _function(x, graph, propagator, self.phase_node, measurement_node)
        sensitivity = -np.abs(grad(_f)(self.phase))
        graph.nodes[self.phase_node]['model'].parameters = [self.phase]
        return sensitivity


