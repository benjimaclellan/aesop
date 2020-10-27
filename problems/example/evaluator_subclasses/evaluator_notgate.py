""" """

import autograd.numpy as np
import numpy
from autograd.numpy.fft import rfft
import scipy.signal as signal
import matplotlib.pyplot as plt

from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_, ifft_shift_
from lib.functions import scale_units


# TODO: figure out how the phase shift causes jumps in the fitness (via incorrect sometimes shifting)

class OpticalNotGate(Evaluator):
    """  """

    def __init__(self, propagator, target, **kwargs):
        super().__init__(**kwargs)

        self.target = (target)

        # self.scale_array = (np.fft.fftshift(
        #     np.linspace(0, len(self.target_f) - 1, len(self.target_f))) / propagator.n_samples).reshape(
        #     (propagator.n_samples, 1))
        #
        # self.target_harmonic_ind = (self.target_harmonic / propagator.df).astype('int')
        # if (self.target_harmonic_ind >= self.target_f.shape[0]):
        #     self.target_harmonic_ind = self.target_f.shape[0]

        self.normalize = False

    def evaluate_graph(self, graph, propagator):
        evaluation_node = graph.get_output_node()  # finds node with no outgoing edges
        graph.propagate(propagator)
        state = graph.measure_propagator(evaluation_node)

        overlap = self.waveform_temporal_similarity(state, propagator)

        score = overlap
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
        target = power_(self.target)
        # shifted = self.shift_function(generated, propagator)
        # similarity_func = self.similarity_l2_norm
        similarity_func = self.similarity_cosine
        similarity = similarity_func(generated, target)
        return similarity

    def shift_function(self, state_power, propagator):
        state_rf = np.fft.fft(state_power, axis=0)

        if (state_rf[self.target_harmonic_ind] == 0):
            return state_power  # no phase shift in this case, and it'll break my lovely gradient otherwise (bit of a hack but...)

        phase = np.angle(state_rf[self.target_harmonic_ind] / self.target_f[self.target_harmonic_ind])
        shift = phase / (self.target_harmonic * propagator.dt)
        state_rf *= np.exp(-1j * shift * self.scale_array)

        shifted = np.abs(np.fft.ifft(state_rf, axis=0))
        return shifted

    def compare(self, graph, propagator):
        evaluation_node = [node for node in graph.nodes if not graph.out_edges(node)][
            0]  # finds node with no outgoing edges

        fig, axs = plt.subplots(2, 1)
        state = graph.measure_propagator(evaluation_node)
        ax = axs[0]
        ax.plot(propagator.t, power_(state), label='Measured State')
        ax.plot(propagator.t, power_(self.target), label='Target State')
        ax.set(xlabel='Time', ylabel='Power a.u.')
        ax.legend()
        scale_units(ax, unit='s', axes=['x'])


        ax = axs[1]
        ls = {'alpha': 0.2}
        ax.plot(propagator.f, np.real(ifft_shift_(fft_(self.target, propagator.dt))), label='Target State - Phase',
                **ls)
        ax.plot(propagator.f, np.real(ifft_shift_(fft_(state, propagator.dt))), label='Measured State - Phase', **ls)
        ax.set(xlabel='Frequency', ylabel='Phase a.u.')
        # ax.legend()
        scale_units(ax, unit='Hz', axes=['x'])
        # ax.plot(propagator.f, psd_(state, propagator.dt, propagator.df), label='Measured State')
        # ax.plot(propagator.f, psd_(self.target, propagator.dt, propagator.df), label='Target State')
        # ax.set(xlabel='Frequency', ylabel='PSD a.u.')
        # ax.legend()
        # scale_units(ax, unit='Hz', axes=['x'])

        ax = axs[1].twinx()
        ls = {'alpha':0.2}
        ax.plot(propagator.f, np.imag(ifft_shift_(fft_(self.target, propagator.dt))), label='Target State - Phase', **ls)
        ax.plot(propagator.f, np.imag(ifft_shift_(fft_(state, propagator.dt))), label='Measured State - Phase', **ls)
        ax.set(xlabel='Frequency', ylabel='Phase a.u.')
        # ax.legend()
        scale_units(ax, unit='Hz', axes=['x'])
        plt.show()

        fig, ax = plt.subplots(1, 1)
        state = graph.measure_propagator(evaluation_node)
        ax.plot(rfspectrum_(state, propagator.dt), label='Measured State')
        ax.plot(rfspectrum_(self.target, propagator.dt), label='Measured State')
        ax.set(xlabel='', ylabel='Power a.u.')
        ax.legend()
        scale_units(ax, unit='Hz', axes=['x'])
        plt.show()
        return
