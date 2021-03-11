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
            p = np.mean(_state)
            return p

        measurement_node = 'sink'
        phase_models = [graph.edges[edge]['model'] for edge in graph.edges if type(graph.edges[edge]['model']) == type(self.phase_model)]

        if len(phase_models) != 1:  # to ensure we are not adding or removing phase shifters. theres should only the one added at the beginning
            # graph.draw()
            raise RuntimeError(f'There is {len(phase_models)} rather than one as expected {phase_models}')

        phase_model = phase_models[0]
        phase = phase_model.parameters[0]

        _f = lambda x: _function(x, graph, propagator, phase_model, measurement_node)
        # sensitivity = -np.abs(grad(_f)(self.phase))
        sensitivity = -np.abs(grad(_f)(phase))
        # phase_model.parameters = [self.phase]
        phase_model.parameters = [phase]
        return sensitivity


