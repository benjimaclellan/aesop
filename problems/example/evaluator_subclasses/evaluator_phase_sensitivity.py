""" """

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_
from lib.functions import scale_units

class PhaseSensitivity(Evaluator):
    """  """

    def __init__(self, propagator, phase=0.0, phase_model=None, **kwargs):
        super().__init__(**kwargs)
        assert phase_model.protected is True

        self.phase_model = phase_model
        self.phase = phase
        self.target = np.ones_like(propagator.t)


    def evaluate_graph(self, graph, propagator):

        def _function(_phase, _graph, _propagator, _phase_model, _measurement_node):
            _phase_model.parameters = [_phase]
            _graph.propagate(_propagator)
            _state = _graph.measure_propagator(_measurement_node)
            p = np.sum(power_(_state))
            return p

        measurement_node = 'sink'
        phase_models = [graph.edges[edge]['model'] for edge in graph.edges if isinstance(graph.edges[edge]['model'], self.phase_model)]
        assert len(phase_models) == 1
        phase_model = phase_models[0]
        _f = lambda x: _function(x, graph, propagator, phase_model, measurement_node)
        sensitivity = -np.abs(grad(_f)(self.phase))
        phase_model.parameters = [self.phase]
        return sensitivity


