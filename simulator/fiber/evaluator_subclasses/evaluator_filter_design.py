""" """

import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_, fft_shift_
from lib.functions import scale_units


class FilterDesign(Evaluator):
    """  """

    def __init__(self, propagator, evaluation_node='sink', **kwargs):
        super().__init__(**kwargs)

        self.fsr = 0.2e12  # Hz
        self.width = 0.02e12  # Hz
        self.duty = self.width/self.fsr
        self.target_transfer = 0.5 * (signal.square(propagator.f / self.fsr + self.duty*np.pi, self.duty) + 1.0)
        self.evaluation_node = evaluation_node

    def evaluate_graph(self, graph, propagator):
        graph.propagate(propagator)
        state = graph.measure_propagator(self.evaluation_node)
        shifted_freq_response = np.abs(fft_shift_(fft_(state, propagator.dt)))
        # score = self.similarity_l1_norm(shifted_freq_response, self.target_transfer)
        # score = self.similarity_l2_norm(shifted_freq_response, self.target_transfer)
        # score = self.similarity_cosine(shifted_freq_response, self.target_transfer)
        # score = self.similarity_mask(shifted_freq_response, self.target_transfer)
        # score = self.similarity_test(shifted_freq_response, self.target_transfer)
        score = self.percent_of_total_power(shifted_freq_response, self.target_transfer)

        # fig, ax = plt.subplots(1, 1)
        # ax.plot(np.abs(fft_shift_(fft_(state, propagator.dt))))
        # ax.plot( self.target_transfer)

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

