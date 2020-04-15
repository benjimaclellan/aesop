""" """

import warnings

from config.config import np
from config.config import sp
import scipy.signal as signal
import matplotlib.pyplot as plt

from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_

class RadioFrequencyWaveformGeneration(Evaluator):
    """  """
    def __init__(self, propagator, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_node = 3  # TODO this is just a placeholder for testing, need dynamic setting

        self.target_harmonic = 12e9  # target pattern repetition in Hz

        self.target = signal.sawtooth(2 * np.pi * self.target_harmonic * propagator.t, 0.5) + 1

        self.target_rf = np.fft.fft(self.target, axis=0)
        self.scale_array = (np.fft.fftshift(
            np.linspace(0, len(self.target_rf) - 1, len(self.target_rf))) / propagator.n_samples).reshape((propagator.n_samples, 1))

        self.target_harmonic_ind = (self.target_harmonic / propagator.df).astype('int')
        self.normalize = False
        return

    def evaluate_graph(self, graph, propagator):
        graph.propagate(propagator)
        state = graph.nodes[self.evaluation_node]['states'][0]
        score = self.waveform_temporal_overlap(state, propagator)
        return score

    def waveform_temporal_overlap(self, state, propagator):
        generated = power_(state)

        if self.normalize:
            generated = generated / np.max(generated)

        shifted = self.shift_function(generated, propagator)

        overlap_integral = np.sum(np.abs(self.target - shifted)) / propagator.n_samples
        return overlap_integral


    def shift_function(self, state_power, propagator):
        state_rf = np.fft.fft(state_power, axis=0)
        phase = np.angle(state_rf[self.target_harmonic_ind] / self.target_rf[self.target_harmonic_ind])

        shift = phase / (self.target_harmonic * propagator.dt)
        state_rf *= np.exp(-1j * shift * self.scale_array)
        shifted = np.abs(np.fft.ifft(state_rf, axis=0))
        return shifted


    def compare(self, graph, propagator):
        fig, ax = plt.subplots(1, 1)
        state = graph.nodes[self.evaluation_node]['states'][0]
        ax.plot(propagator.t, power_(state), label='Measured State')
        ax.plot(propagator.t, self.target, label='Target State')
        ax.set(xlabel='Time (s)', ylabel='Power a.u.')
        ax.legend()
        plt.show()
        return

