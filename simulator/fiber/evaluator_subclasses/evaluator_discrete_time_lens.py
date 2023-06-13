""" """

import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_, fft_shift_
from lib.functions import scale_units


class DiscreteTimeLens(Evaluator):
    """  """

    def __init__(self, propagator, evaluation_node='sink', **kwargs):
        super().__init__(**kwargs)

        self.fsr = 0.1235e11  # Hz
        self.phase = 1.9
        self.target_psd_osc = 2.0 * np.cos(propagator.f / self.fsr + self.phase)**2
        self.evaluation_node = evaluation_node

    def evaluate_graph(self, graph, propagator):
        graph.propagate(propagator)
        input_state = graph.measure_propagator((0, 1, 0))
        target_psd = self.target_psd_osc * np.abs(fft_shift_(fft_(input_state, propagator.dt)))

        state = graph.measure_propagator(self.evaluation_node)
        shifted_freq_response = np.abs(fft_shift_(fft_(state, propagator.dt)))

        shifted_freq_response = shifted_freq_response / np.max(target_psd)
        target_psd = target_psd / np.max(target_psd)

        # score = self.similarity_l1_norm(shifted_freq_response, target_psd)
        score = self.similarity_l2_norm(shifted_freq_response, target_psd)

        # score = self.similarity_cosine(shifted_freq_response, target_psd)
        # score = self.similarity_mask(shifted_freq_response, target_psd)
        # score = self.similarity_test(shifted_freq_response, target_psd)
        # score = self.percent_of_total_power(shifted_freq_response, target_psd)
        # plt.plot(shifted_freq_response)
        # plt.plot(target_psd)
        # plt.show()

        max_ = np.max(shifted_freq_response[propagator.n_samples//4:3*propagator.n_samples//4])
        min_ = np.min(shifted_freq_response[propagator.n_samples//4:3*propagator.n_samples//4])
        vis = (max_ - min_) / (max_ + min_)
        score = score / vis  #(np.abs(2 - np.max(shifted_freq_response)) + np.abs(np.min(shifted_freq_response)))

        return score

    @staticmethod
    def similarity_cosine(x_, y_):
        return np.sum(x_ * y_) / (np.sum(np.sqrt(np.power(x_, 2))) * np.sum(np.sqrt(np.power(y_, 2))))

    def similarity_mask(self, x_, y_):
        return np.sum(np.power(x_ - y_, 2) * y_) + self.duty*np.sum(np.power(x_ - y_, 2) * 1-y_)

    def similarity_test(self, x_, y_):
        return -np.sum(x_ * y_)

    def percent_of_total_power(self, x_, y_):
        return -np.sum(x_ * y_) / np.sum(x_)

    @staticmethod
    def similarity_l1_norm(x_, y_):
        return np.sum(np.abs(x_ - y_))

    @staticmethod
    def similarity_l2_norm(x_, y_):
        # return np.sum(np.power(x_ - y_, 2))
        return np.sum(np.power(x_ - y_, 2))

