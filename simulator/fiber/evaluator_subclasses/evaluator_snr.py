""" """

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_
from lib.functions import scale_units

from simulator.fiber.assets.additive_noise import AdditiveNoise


class SignalNoiseRatio(Evaluator):
    """
    Uses the Signal-to-Noise ratio (SNR) as the objective function
    The input graph is simulated once with noise injection and once without
    """

    def __init__(self, propagator, **kwargs):
        super().__init__(**kwargs)

        self.target = np.ones_like(propagator.t)

    def evaluate_graph(self, graph, propagator):

        measurement_node = 'sink'
        AdditiveNoise.simulate_with_noise = False
        graph.propagate(propagator)
        state_signal = np.abs(graph.measure_propagator(measurement_node))

        AdditiveNoise.simulate_with_noise = True
        graph.propagate(propagator)
        state_noise = np.abs(graph.measure_propagator(measurement_node))

        noise = state_signal - state_noise
        # snr = -np.mean(noise)
        snr = np.mean(noise/state_signal)
        return snr


