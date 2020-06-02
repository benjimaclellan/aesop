import pytest
import autograd.numpy as np
from autograd.test_util import check_grads
import numpy.polynomial.polynomial as poly
import matplotlib.patches as patches
import matplotlib.pyplot as plt

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


def test_norm1_weight0():
    pass


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


def test_BER_differentiation(propagator_6samples, bit_seq_2bits):
    evaluator = BERCaseEvaluator(propagator_6samples, bit_seq_2bits, 3)
    check_grads(evaluator.evaluate_graph)

