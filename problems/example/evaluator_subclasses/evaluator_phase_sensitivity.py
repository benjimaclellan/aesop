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


    def evaluate_graph(self, graph, propagator):
        evaluation_node = graph.get_output_node()  # finds node with no outgoing edges

        def _function(_phase, _graph, _node):
            _graph.nodes[_node]['model'].parameters = [_phase]
            _graph.propagate
            graph.propagate(propagator)
            state = graph.measure_propagator(evaluation_node)
            p = np.sum(power_(state))
            return p
        _f = lambda x: _function(x, graph, self.phase_node)

        sensitivity = np.abs(grad(_f)(self.phase))
        return -sensitivity


