import pytest
import autograd.numpy as np
from autograd.test_util import check_grads
import numpy.polynomial.polynomial as poly
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math

from problems.example.evaluator_subclasses.weighted_evaluators.case_evaluator_norm import NormCaseEvaluator
from problems.example.evaluator_subclasses.weighted_evaluators.case_evaluator_BER import BERCaseEvaluator
from problems.example.evaluator_subclasses.max_eye_evaluator import MaxEyeEvaluator
from problems.example.assets.propagator import Propagator


@pytest.fixture(scope='module')
def default_2bits_3samplesbit_mock():
    return np.array([0.5, 1, 0.7, 0.4, 0.2, 0.41])


@pytest.fixture(scope='module')
def bit_seq_2bits():
    return np.array([1, 0], dtype=bool)


@pytest.fixture(scope='module')
def propagator_6samples():
    return Propagator(n_samples=6, window_t=6, central_wl=1)


def test_norm1_weight0(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = NormCaseEvaluator(propagator_6samples, bit_seq_2bits, 3,
                                   weighting_exponent=0, mock_graph_for_testing=True)
    sum = (0.5 + 0.3 + 0.4 + 0.2 + 0.41) / 6
    assert math.isclose(evaluator.evaluate_graph(default_2bits_3samplesbit_mock), sum)


def test_norm1_weight2(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = NormCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, mock_graph_for_testing=True)
    sum = (2/3) * (1/4 * 0.5 + 1/4 * 0.3 + 1/4 * 0.4 + 1 * 0.2 + 1/4 * 0.41)
    assert math.isclose(evaluator.evaluate_graph(default_2bits_3samplesbit_mock), sum / 2)


def test_norm2_weight4(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = NormCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, norm=2,
                                  weighting_exponent=4, mock_graph_for_testing=True)
    sum = (8/9)**2 * ((1/16 * 0.5)**2 + (1/16 * 0.3)**2 + (1/16 * 0.4)**2 + (1 * 0.2)**2 + (1/16 * 0.41)**2)
    weighed_norm = np.sqrt(sum)
    assert math.isclose(evaluator.evaluate_graph(default_2bits_3samplesbit_mock), weighed_norm / 2)


def test_norm_differentation(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = NormCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, norm=2, mock_graph_for_testing=True)
    check_grads(evaluator.evaluate_graph)(default_2bits_3samplesbit_mock)


def test_BER_weight1(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    # weight 1, thresh default
    evaluator = BERCaseEvaluator(propagator_6samples, bit_seq_2bits, 3,
                                 weighting_exponent=1,
                                 mock_graph_for_testing=True)
    assert np.abs(evaluator.evaluate_graph(default_2bits_3samplesbit_mock) - 1 / 2) < 0.05


def test_BER_weight3_threshpoint2(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    # weight 3, thresh 0.2
    evaluator = BERCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, thresh=0.2,
                                 weighting_exponent=3, mock_graph_for_testing=True)
    assert np.abs(evaluator.evaluate_graph(default_2bits_3samplesbit_mock) - 1 / 2) < 0.05


def test_BER_weight3_threshpoint25(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    # weight 3, thresh 0.25
    evaluator = BERCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, thresh=0.25,
                                 weighting_exponent=3, mock_graph_for_testing=True)
    assert np.abs(evaluator.evaluate_graph(default_2bits_3samplesbit_mock)) < 0.05


def test_BER_differentiation(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = BERCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, mock_graph_for_testing=True)
    check_grads(evaluator.evaluate_graph)(default_2bits_3samplesbit_mock)

