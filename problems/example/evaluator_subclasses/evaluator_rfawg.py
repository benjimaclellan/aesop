""" """

import warnings

import autograd.numpy as np

import scipy.signal as signal

import matplotlib.pyplot as plt

from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_
from lib.functions import scale_units

# TODO: figure out how the phase shift causes jumps in the fitness (via incorrect sometimes shifting)

class RadioFrequencyWaveformGeneration(Evaluator):
    """  """
    def __init__(self, propagator, **kwargs):
        super().__init__(**kwargs)
        # self.evaluation_node = 3  # TODO this is just a placeholder for testing, need dynamic setting

        self.target_harmonic = 12e9  # target pattern repetition in Hz

        self.target = signal.sawtooth(2 * np.pi * self.target_harmonic * propagator.t, 0.5) + 1

        self.target_rf = np.fft.fft(self.target, axis=0)
        self.scale_array = (np.fft.fftshift(
            np.linspace(0, len(self.target_rf) - 1, len(self.target_rf))) / propagator.n_samples).reshape((propagator.n_samples, 1))

        # self.target_harmonic_ind = (np.rint(self.target_harmonic / propagator.df)).astype('int')
        self.target_harmonic_ind = (self.target_harmonic / propagator.df).astype('int')
        if (self.target_harmonic_ind >= self.target_rf.shape[0]):
            self.target_harmonic_ind = self.target_rf.shape[0]
        
        self.normalize = False

    def evaluate_graph(self, graph, propagator):
        evaluation_node = graph.get_output_node()  # finds node with no outgoing edges
        graph.propagate(propagator)
        state = graph.measure_propagator(evaluation_node)
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
        
        if (state_rf[self.target_harmonic_ind] == 0):
            return state_power # no phase shift in this case, and it'll break my lovely gradient otherwise (bit of a hack but...)
        
        phase = np.angle(state_rf[self.target_harmonic_ind] / self.target_rf[self.target_harmonic_ind])

        shift = phase / (self.target_harmonic * propagator.dt)
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







class PulseShaping(Evaluator):
    """  """
    def __init__(self, propagator, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_node = 3  # TODO this is just a placeholder for testing, need dynamic setting

        train = False
        t_rep = 500e-12 # 100 MHz
        self.pulse_width = 25e-12  # pulse width target in s

        if train:
            duty_cycle = self.pulse_width/t_rep
            self.target = (signal.square(2*np.pi * propagator.t/t_rep, duty_cycle) + 1)/2
        else:
            self.target = np.zeros_like(propagator.t)
            self.target[abs(propagator.t) < self.pulse_width/2 ] = 1

        self.normalize = True
        return

    def evaluate_graph(self, graph, propagator):
        graph.propagate(propagator)
        state = graph.nodes[self.evaluation_node]['states'][0]
        score = self.waveform_temporal_overlap(state, propagator)
        return score

    def waveform_temporal_overlap(self, state, propagator):
        generated = power_(state)
        return -np.sum(generated * self.target) / np.sum(generated * np.logical_not(self.target).astype('float'))
        # if self.normalize:
        #     generated = generated / np.max(generated)
        # overlap_integral = np.sum(np.abs(self.target - generated)) / propagator.n_samples
        # return overlap_integral

    def compare(self, graph, propagator):
        fig, ax = plt.subplots(1, 1)
        state = graph.nodes[self.evaluation_node]['states'][0]
        generated = power_(state)
        if self.normalize:
            generated = generated / np.max(generated)
        ax.plot(propagator.t, generated, label='Measured State')
        ax.plot(propagator.t, self.target, label='Target State')
        ax.set(xlabel='Time', ylabel='Power a.u.')
        ax.legend()
        scale_units(ax, unit='s', axes=['x'])
        plt.show()
        return

