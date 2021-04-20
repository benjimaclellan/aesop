""" """

import autograd.numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from numpy.fft import rfft

from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_, ifft_shift_, fft_shift_
from lib.functions import scale_units

# TODO: figure out how the phase shift causes jumps in the fitness (via incorrect sometimes shifting)

class RadioFrequencyWaveformGeneration(Evaluator):
    """  """
    def __init__(self, propagator,
                 target_harmonic=12e9,
                 target_amplitude=0.02,  # V - ensure we have a PD as the final component
                 target_waveform='saw',
                 evaluation_node_edge='sink',
                 **kwargs):
        super().__init__(**kwargs)
        self.evaluation_node_edge = evaluation_node_edge
        self.target_harmonic = target_harmonic  # target pattern repetition in Hz
        self.target_amplitude = target_amplitude

        if type(target_waveform) == str:
            if target_waveform == 'saw':
                waveform = 0.5 * (signal.sawtooth(2 * np.pi * self.target_harmonic * propagator.t, 0.0) + 1)
            if target_waveform == 'other-saw':
                waveform = 0.5 * (signal.sawtooth(2 * np.pi * self.target_harmonic * propagator.t, 1.0) + 1)
            elif target_waveform == 'tri':
                waveform = 0.5 * (signal.sawtooth(2 * np.pi * self.target_harmonic * propagator.t, 0.5) + 1)
            elif target_waveform == 'square':
                waveform = 0.5 * (signal.square(2 * np.pi * self.target_harmonic * propagator.t, 0.5) + 1)
            else:
                raise NotImplementedError(f'{target_waveform} is not a valid target waveform')

            self.target = target_amplitude * waveform

        elif type(target_waveform) is np.ndarray:
            self.target = target_waveform

        self.target_f = fft_(self.target, propagator.dt)
        self.target_rf = fft_(self.target, propagator.dt)

        self.scale_array = ifft_shift_(propagator.f) / self.target_harmonic

        self.target_harmonic_ind = (self.target_harmonic / propagator.df).astype('int') + 1
        if (self.target_harmonic_ind >= self.target_f.shape[0]):
            self.target_harmonic_ind = self.target_f.shape[0]
        
        self.normalize = False

    def evaluate_graph(self, graph, propagator):
        state = graph.measure_propagator(self.evaluation_node_edge)

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
        return np.mean(np.power(x_ - y_, 2))

    def waveform_temporal_similarity(self, voltage, propagator):
        generated = voltage

        shifted = self.shift_function(generated, propagator)
        similarity_func = self.similarity_l2_norm
        similarity = similarity_func(shifted, self.target) / self.target_amplitude ** 2
        return similarity

    def shift_function(self, state_power, propagator):
        state_rf = fft_(state_power, propagator.dt)
        # if (state_rf[self.target_harmonic_ind] == 0):
        #     return state_power # no phase shift in this case, and it'll break my lovely gradient otherwise (bit of a hack but...)

        phase = np.angle(state_rf[self.target_harmonic_ind] / self.target_f[self.target_harmonic_ind])

        # plt.figure()
        # plt.plot(propagator.f, np.abs(state_rf), color='green')
        # plt.plot(propagator.f, np.abs(self.target_f), color='red')
        # plt.scatter(propagator.f[self.target_harmonic_ind], np.abs(self.target_f)[self.target_harmonic_ind])
        # plt.scatter(propagator.f[self.target_harmonic_ind], np.abs(state_rf)[self.target_harmonic_ind])
        # print(phase)

        shifted = np.abs(ifft_(np.exp(1j * phase * self.scale_array) * fft_(state_power, propagator.dt), propagator.dt))
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