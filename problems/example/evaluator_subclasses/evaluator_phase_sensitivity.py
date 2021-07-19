""" """

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, grad_named
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




class SecondOrderDifferentialSensitivity(Evaluator):
    """  
    Experimental evaluator class which will be a somewhat dummy function between the average optical power 
    and the partial phase derivative terms of two phase shifters
    """

    def __init__(self, propagator, phase1=0.0, phase2=0.0, phase_model1=None, phase_model2=None, **kwargs):
        super().__init__(**kwargs)
        for phase_model in [phase_model1, phase_model2]:
            assert phase_model.protected is True

        self.phase_model1 = phase_model1
        self.phase_model2 = phase_model2
        self.phase1 = phase1
        self.phase2 = phase2
        self.target = np.ones_like(propagator.t)


    def evaluate_graph(self, graph, propagator):
        def _function_phase_both(_phase1, _phase2, _graph, _propagator, _phase_model1, _phase_model2, _measurement_node):
            _phase_model1.parameters = [_phase1]
            _phase_model2.parameters = [_phase2]
            _graph.propagate(_propagator)
            _state = _graph.measure_propagator(_measurement_node)
            p = np.mean(_state)
            return p

        measurement_node = 'sink'
        phase_models = [graph.edges[edge]['model'] for edge in graph.edges if type(graph.edges[edge]['model']) == type(self.phase_model1)]

        if not len(phase_models) >= 2:  # make sure this is okay, we can add phase models, but need at least these two
            # graph.draw()
            raise RuntimeError(f'There is {len(phase_models)} rather than one as expected {phase_models}')

        phase_model1 = self.phase_model1
        phase1 = phase_model1.parameters[0]

        phase_model2 = self.phase_model2
        phase2 = phase_model2.parameters[0]

        _f = lambda phase1, phase2: _function_phase_both(phase1, phase2, graph, propagator, phase_model1, phase_model2, measurement_node)

        term1 = grad(_f, 0)
        term2 = grad(grad(_f, 0), 1)

        loss = np.abs(term1(phase1, phase2) + term2(phase1, phase2))

        # make sure we set back to the right datatype - hacky but works
        phase_model1.parameters = [phase1]
        phase_model2.parameters = [phase2]
        return loss

