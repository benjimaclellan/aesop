""" """

import autograd.numpy as np
import matplotlib.pyplot as plt

from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_
from lib.functions import scale_units


class PulseRepetition(Evaluator):
    """  """

    def __init__(self, propagator, **kwargs):
        super().__init__(**kwargs)

        self.target_pulse_rep_t = 5e-9  # target pattern repetition time in s
        self.target_pulse_width = 0.1e-9 # pulse width in s
        self.target_pulse_rep_f = 1/self.target_pulse_rep_t  # target reprate in Hz

        self.target = 0.5 * power_(get_pulse_train(propagator.t, self.target_pulse_rep_t, self.target_pulse_width, pulse_shape='gaussian'))

        self.target_f = np.fft.fft(self.target, axis=0)
        self.target_rf = np.fft.rfft(self.target, axis=0)

        self.scale_array = (np.fft.fftshift(
            np.linspace(0, len(self.target_f) - 1, len(self.target_f))) / propagator.n_samples).reshape(
            (propagator.n_samples, 1))

        self.target_harmonic_ind = (self.target_pulse_rep_f / propagator.df).astype('int')
        if (self.target_harmonic_ind >= self.target_f.shape[0]):
            self.target_harmonic_ind = self.target_f.shape[0]


    def evaluate_graph(self, graph, propagator):
        evaluation_node = graph.get_output_node()  # finds node with no outgoing edges

        graph.propagate(propagator)
        state = graph.measure_propagator(evaluation_node)

        num_nodes = len(graph.nodes) - 2
        power_loss_approx = (0.95) ** num_nodes  # TODO make more valid loss calculation

        overlap = self.waveform_temporal_similarity(state, propagator)

        score = overlap * (power_loss_approx)

        return score

    @staticmethod
    def similarity_cosine(x_, y_):
        return np.sum(x_ * y_) / (np.sum(np.sqrt(np.power(x_, 2))) * np.sum(np.sqrt(np.power(y_, 2))))

    @staticmethod
    def similarity_l1_norm(x_, y_):
        return np.sum(np.abs(x_ - y_))

    @staticmethod
    def similarity_l2_norm(x_, y_):
        return np.sum(np.power(x_ - y_, 2))

    def waveform_temporal_similarity(self, state, propagator):
        generated = power_(state)
        shifted = self.shift_function(generated, propagator)
        similarity_func = self.similarity_l2_norm
        similarity = -np.abs(np.fft.rfft(generated, axis=0))[self.target_harmonic_ind]
        # similarity = similarity_func(np.abs(np.fft.rfft(generated, axis=0)), np.abs(self.target_rf))
        # similarity = similarity_func(shifted, self.target)
        # similarity = similarity_func(generated, self.target)
        return similarity

    def shift_function(self, state_power, propagator):
        state_rf = np.fft.fft(state_power, axis=0)

        if (state_rf[self.target_harmonic_ind] == 0):
            return state_power  # no phase shift in this case, and it'll break my lovely gradient otherwise (bit of a hack but...)

        phase = np.angle(state_rf[self.target_harmonic_ind] / self.target_f[self.target_harmonic_ind])
        shift = phase / (self.target_pulse_rep_f * propagator.dt)
        state_rf *= np.exp(-1j * shift * self.scale_array)

        shifted = np.abs(np.fft.ifft(state_rf, axis=0))
        return shifted


    def compare(self, graph, propagator):
        evaluation_node = [node for node in graph.nodes if not graph.out_edges(node)][0]  # finds node with no outgoing edges

        fig, ax = plt.subplots(1, 1)
        state = graph.measure_propagator(evaluation_node)
        ax.plot(propagator.t, power_(state), label='Measured State')
        ax.plot(propagator.t, self.target, label='Target State')
        ax.set(xlabel='Time', ylabel='Power a.u.')
        ax.legend()
        scale_units(ax, unit='s', axes=['x'])
        plt.show()

        fig, ax = plt.subplots(1, 1)
        state = graph.measure_propagator(evaluation_node)
        ax.plot(rfspectrum_(state, propagator.dt), label='Measured State')
        ax.set(xlabel='', ylabel='Power a.u.')
        ax.legend()
        scale_units(ax, unit='Hz', axes=['x'])
        plt.show()
        return


