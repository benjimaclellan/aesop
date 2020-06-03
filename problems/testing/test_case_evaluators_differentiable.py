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


GRAPHICAL_TESTING = True


@pytest.fixture(scope='module')
def default_2bits_3samplesbit_mock():
    return np.array([0.5, 1, 0.7, 0.4, 0.2, 0.41])


@pytest.fixture(scope='module')
def default_6bits_7samplesbit_mock():
    return np.array(
        [0.1, 0.2, 0.1, 0.05, 0.07, 0.12, 0.11,
         0.12, 0.06, 0.03, 0.09, 0.1, 0.18, 0.26,
         0.65, 0.77, 0.95, 0.97, 0.94, 1, 0.98,
         0.87, 0.89, 0.92, 0.99, 0.83, 0.76, 0.55,
         0.38, 0.22, 0.12, 0.05, 0.08, 0.18, 0.25, 
         0.69, 0.75, 0.89, 0.95, 0.97, 0.85, 0.7])


@pytest.fixture(scope='module')
def bit_seq_6bits():
    return np.array([0, 0, 1, 1, 0, 1], dtype=bool)


@pytest.fixture(scope='module')
def bit_seq_2bits():
    return np.array([1, 0], dtype=bool)


@pytest.fixture(scope='module')
def propagator_6samples():
    return Propagator(n_samples=6, window_t=6, central_wl=1)


@pytest.fixture(scope='module')
def propagator_42samples():
    return Propagator(n_samples=42, window_t=1, central_wl=1)


@pytest.fixture(scope='module')
def propagator_single_sample():
    return Propagator(n_samples=1, window_t=1, central_wl=1)


@pytest.fixture(scope='module')
def max_eye_evaluator_basic(propagator_6samples, bit_seq_2bits):
    return MaxEyeEvaluator(propagator_6samples, bit_seq_2bits, 3,
                           mock_graph_for_testing=True, graphical_testing=GRAPHICAL_TESTING)


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


# def test_norm_differentation(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
#     evaluator = NormCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, norm=2, mock_graph_for_testing=True)
#     check_grads(evaluator.evaluate_graph)(default_2bits_3samplesbit_mock)


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


# def test_BER_differentiation(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
#     evaluator = BERCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, mock_graph_for_testing=True)
#     check_grads(evaluator.evaluate_graph)(default_2bits_3samplesbit_mock)

@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='graphical testing disabled')
def test_max_eye_graphical_simple(max_eye_evaluator_basic, default_2bits_3samplesbit_mock):
    max_eye_evaluator_basic.evaluate_graph(default_2bits_3samplesbit_mock)


@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='graphical testing disabled')
def test_max_eye_graphical_longer(propagator_42samples, bit_seq_6bits,
                                  default_6bits_7samplesbit_mock):
    evaluator = MaxEyeEvaluator(propagator_42samples, bit_seq_6bits, 7,
                                mock_graph_for_testing=True,
                                graphical_testing=GRAPHICAL_TESTING)
    evaluator.evaluate_graph(default_6bits_7samplesbit_mock)


def test_max_eye_0(propagator_single_sample):
    bit_seq = np.array([1])
    evaluator = MaxEyeEvaluator(propagator_single_sample, bit_seq, 1,
                                mock_graph_for_testing=True)
    mock = np.array([1])
    s_squared = 0.5**2 / 0.3**2 + 0.5**2 / 0.15**2
    print(f'test s_squared{s_squared}')
    score = 1 / (np.pi * 0.3 * 0.15 * s_squared)
    assert math.isclose(evaluator.evaluate_graph(mock), score)