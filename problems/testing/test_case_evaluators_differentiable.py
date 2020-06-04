import pytest
import autograd.numpy as np
from autograd.test_util import check_grads
import numpy.polynomial.polynomial as poly
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math
import seaborn as sns

from problems.example.evaluator_subclasses.weighted_evaluators.case_evaluator_norm import NormCaseEvaluator
from problems.example.evaluator_subclasses.weighted_evaluators.case_evaluator_BER import BERCaseEvaluator
from problems.example.evaluator_subclasses.max_eye_evaluator import MaxEyeEvaluator
from problems.example.assets.propagator import Propagator

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper
from problems.example.graph import Graph

from lib.analysis.hessian import get_hessian, get_scaled_hessian, function_wrapper

"""
TODO:
1. Figure out why reusing the same graph twice breaks everything: bug came up where the second time a given
   graph was used to compute the hessian, it did not work
2. Address all the Bad Things happening with BER
    UPDATE: it's not with BER, I think it's with any multiple of 2 smh
    UPDATE: if <bit_seq_length> * <bit_num> divides <propagator length>, it fails
    UPDATE: it works if we axe the phase shift (I mean it breaks the effect of things, but it
            doesn't throw all our errors)
    UPDATE: ok so it looks like the problem is something to do with the phase,
            and self._target_rf[target_harmonic_index] == 0 (which as the fundamental freq, it really shouldn't)

This is our problem:
For <bit_seq_length> * <bit_num> divides <propagator length>:
target_rf[target_harmonic_index - 2]: [0.+0.j]
target_rf[target_harmonic_index - 1]: [0.+0.j]
target_rf[target_harmonic_index]: [0.+0.j]
target_rf[target_harmonic_index + 1]: [2048.-4944.30937574j]
target_rf[target_harmonic_index + 2]: [0.+0.j]

^^ divides by 0 with target_rf[target_harmonic_index] and breaks

For other cases:
target_rf[target_harmonic_index - 2]: [282.86380162+488.77943451j]
target_rf[target_harmonic_index - 1]: [452.2320106+782.13390554j]
target_rf[target_harmonic_index]: [1129.70492961+1955.55154988j]
target_rf[target_harmonic_index + 1]: [-2257.65983154-3911.53619236j]
target_rf[target_harmonic_index + 2]: [-563.97749247-977.99220141j]

^^ divides by not 0, so it's still kind of chill

Solution: rounded instead of truncating on picking target_harmonic_index

TODO: normalize after phase shift?
"""

GRAPHICAL_TESTING = False
EXCLUDE_LOCKED = True

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
         0.69, 0.75, 0.89, 0.95, 0.97, 0.85, 0.7]
    )


@pytest.fixture(scope='module')
def bit_seq_6bits():
    return np.array([0, 0, 1, 1, 0, 1], dtype=bool)


@pytest.fixture(scope='module')
def bit_seq_2bits():
    return np.array([1, 0], dtype=bool)


@pytest.fixture(scope='module')
def bit_seq_3bits():
    return np.array([0, 0, 1], dtype=bool)


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


@pytest.fixture(scope='function')
def default_graph():
    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             1:PhaseModulator(parameters_from_name={'depth':9.87654321, 'frequency':12e9}),
             2:WaveShaper(),
             3:MeasurementDevice()}
    edges = [(0,1, CorningFiber(parameters=[0])),
             (1,2, CorningFiber(parameters=[0])),
             (2,3)]

    graph = Graph(nodes, edges, propagate_on_edges = True)
    graph.assert_number_of_edges()
    return graph


@pytest.fixture(scope='module')
def default_propagator():
    return Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)


def test_norm1_weight0(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = NormCaseEvaluator(propagator_6samples, bit_seq_2bits, 3,
                                   weighting_exponent=0, mock_graph_for_testing=True, phase_shift=False)
    sum = (0.5 + 0.3 + 0.4 + 0.2 + 0.41) / 6
    assert math.isclose(evaluator.evaluate_graph(default_2bits_3samplesbit_mock, propagator_6samples), sum)


def test_norm1_weight2(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = NormCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, mock_graph_for_testing=True,
                                  phase_shift=False)
    sum = (2/3) * (1/4 * 0.5 + 1/4 * 0.3 + 1/4 * 0.4 + 1 * 0.2 + 1/4 * 0.41)
    assert math.isclose(evaluator.evaluate_graph(default_2bits_3samplesbit_mock, propagator_6samples), sum / 2)


def test_norm2_weight4(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = NormCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, norm=2,
                                  weighting_exponent=4, mock_graph_for_testing=True, phase_shift=False)
    sum = (8/9)**2 * ((1/16 * 0.5)**2 + (1/16 * 0.3)**2 + (1/16 * 0.4)**2 + (1 * 0.2)**2 + (1/16 * 0.41)**2)
    weighed_norm = np.sqrt(sum)
    assert math.isclose(evaluator.evaluate_graph(default_2bits_3samplesbit_mock, propagator_6samples), weighed_norm / 2)


@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='not doing graphical testing')
def test_BER_differentiation(default_graph, default_propagator, bit_seq_2bits):
    evaluator = BERCaseEvaluator(default_propagator, bit_seq_2bits, 2, phase_shift=False)

    exclude_locked = EXCLUDE_LOCKED
    info = default_graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'],
                                                                  get_location_indices=True,
                                                                  exclude_locked=exclude_locked)
    hessian = get_hessian(default_graph, default_propagator, evaluator, exclude_locked=exclude_locked)

    H0 = hessian(np.array(info['parameters']))
    print(f'H0: {H0}, H0 type: {type(H0)}')

    fig, ax = plt.subplots()
    sns.heatmap(H0)
    ax.set(xticks = list(range(len(info['parameters']))), yticks = list(range(len(info['parameters']))))
    ax.set_xticklabels(info['parameter_names'], rotation=45, ha='center')
    ax.set_yticklabels(info['parameter_names'], rotation=45, ha='right')
    plt.title('BER Hessian')
    plt.show()
    
    # func = function_wrapper(default_graph, default_propagator, evaluator, exclude_locked=exclude_locked)
    # check_grads(func)(default_2bits_3samplesbit_mock)


def test_BER_weight1(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    # weight 1, thresh default
    evaluator = BERCaseEvaluator(propagator_6samples, bit_seq_2bits, 3,
                                 weighting_exponent=1,
                                 mock_graph_for_testing=True,
                                 phase_shift=False)
    assert np.abs(evaluator.evaluate_graph(default_2bits_3samplesbit_mock, propagator_6samples) - 1 / 2) < 0.05


def test_BER_weight3_threshpoint2(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    # weight 3, thresh 0.2
    evaluator = BERCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, thresh=0.2,
                                 weighting_exponent=3, mock_graph_for_testing=True,
                                 phase_shift=False)
    assert np.abs(evaluator.evaluate_graph(default_2bits_3samplesbit_mock, propagator_6samples) - 1 / 2) < 0.05


def test_BER_weight3_threshpoint25(propagator_6samples, bit_seq_2bits, default_2bits_3samplesbit_mock):
    # weight 3, thresh 0.25
    evaluator = BERCaseEvaluator(propagator_6samples, bit_seq_2bits, 3, thresh=0.25,
                                 weighting_exponent=3, mock_graph_for_testing=True,
                                 phase_shift=False)
    assert np.abs(evaluator.evaluate_graph(default_2bits_3samplesbit_mock, propagator_6samples)) < 0.05


@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='not doing graphical testing')
def test_norm_differentation(default_graph, default_propagator, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = NormCaseEvaluator(default_propagator, bit_seq_2bits, 4, norm=1)
    exclude_locked = EXCLUDE_LOCKED
    info = default_graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'],
                                                                  get_location_indices=True,
                                                                  exclude_locked=exclude_locked)
    hessian = get_hessian(default_graph, default_propagator, evaluator, exclude_locked=exclude_locked)

    H0 = hessian(np.array(info['parameters']))
    print(f'H0: {H0}, H0 type: {type(H0)}')

    fig, ax = plt.subplots()
    sns.heatmap(H0)
    ax.set(xticks = list(range(len(info['parameters']))), yticks = list(range(len(info['parameters']))))
    ax.set_xticklabels(info['parameter_names'], rotation=45, ha='center')
    ax.set_yticklabels(info['parameter_names'], rotation=45, ha='right')
    plt.title('Norm Hessian')
    plt.show()
    
    # func = function_wrapper(default_graph, default_propagator, evaluator, exclude_locked=exclude_locked)
    # check_grads(func)(default_2bits_3samplesbit_mock)


@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='graphical testing disabled')
def test_max_eye_graphical_simple(max_eye_evaluator_basic, default_2bits_3samplesbit_mock):
    max_eye_evaluator_basic.evaluate_graph(default_2bits_3samplesbit_mock, propagator_6samples)


@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='graphical testing disabled')
def test_max_eye_graphical_longer(propagator_42samples, bit_seq_6bits,
                                  default_6bits_7samplesbit_mock):
    evaluator = MaxEyeEvaluator(propagator_42samples, bit_seq_6bits, 7,
                                mock_graph_for_testing=True,
                                graphical_testing=GRAPHICAL_TESTING)
    evaluator.evaluate_graph(default_6bits_7samplesbit_mock, propagator_6samples)


def test_max_eye_0(propagator_single_sample):
    bit_seq = np.array([1])
    evaluator = MaxEyeEvaluator(propagator_single_sample, bit_seq, 1,
                                mock_graph_for_testing=True)
    mock = np.array([1])
    s_squared = 0.5**2 / 0.3**2 + 0.5**2 / 0.15**2
    score = 1 / (np.pi * 0.3 * 0.15 * s_squared)
    assert math.isclose(evaluator.evaluate_graph(mock, propagator_single_sample), score)


@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='not doing graphical testing')
def test_maxEye_differentation(default_graph, default_propagator, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = MaxEyeEvaluator(default_propagator, bit_seq_2bits, 4)
    exclude_locked = EXCLUDE_LOCKED
    info = default_graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'],
                                                                  get_location_indices=True,
                                                                  exclude_locked=exclude_locked)
    hessian = get_hessian(default_graph, default_propagator, evaluator, exclude_locked=exclude_locked)

    H0 = hessian(np.array(info['parameters']))
    print(f'H0: {H0}, H0 type: {type(H0)}')

    fig, ax = plt.subplots()
    sns.heatmap(H0)
    ax.set(xticks = list(range(len(info['parameters']))), yticks = list(range(len(info['parameters']))))
    ax.set_xticklabels(info['parameter_names'], rotation=45, ha='center')
    ax.set_yticklabels(info['parameter_names'], rotation=45, ha='right')
    plt.title('Max Eye Hessian')
    plt.show()

@pytest.mark.skip
def test_maxEye_checkGrad(default_graph, default_propagator, bit_seq_2bits, default_2bits_3samplesbit_mock):
    evaluator = MaxEyeEvaluator(default_propagator, bit_seq_2bits, 4)
    func = function_wrapper(default_graph, default_propagator, evaluator, exclude_locked=EXCLUDE_LOCKED)
    check_grads(func)(default_2bits_3samplesbit_mock)

@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='not doing graphical testing')
def test_phase_shift_integer_bitSeq():
    # just inspect manually to see that it makes sense
    # TODO: add a test, that should be pretty easy
    # integer bit sequence means that the bit sequence fits into the propagator an integer # of times
    BIT_WIDTH = 3
    seq_bit = np.array([0, 0, 1, 1])
    propagator = Propagator(n_samples= 16 * BIT_WIDTH, window_t=16, central_wl=1)
    shifted_seq_bit = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    pre_shift_output = np.reshape(np.resize(np.repeat(shifted_seq_bit, BIT_WIDTH), propagator.n_samples),
                                (propagator.n_samples, 1))
    evaluator = NormCaseEvaluator(propagator, seq_bit, BIT_WIDTH)
    print('NO NOISE:')
    print(f'target: {evaluator._target}')
    print(f'pre_shift_output: {pre_shift_output}')
    print(f'post_shift_output: {evaluator._align_phase(pre_shift_output)}')
    
    noisy_pre_shift_output = pre_shift_output + np.random.normal(0, 0.05, pre_shift_output.shape[0])
    print('WITH NOISE:')
    print(f'pre_shift_output: {noisy_pre_shift_output}')
    print(f'post_shift_output: {evaluator._align_phase(noisy_pre_shift_output)}')

    assert False # otherwise won't print outputs

@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='not doing graphical testing')
def test_phase_shift_nonInteger_bitSeq():
    # just inspect to see that it makes sense
    # non-integer is because the bit sequence does not fit into the propagator an int # of times
    BIT_WIDTH = 3
    seq_bit = np.array([0, 0, 1, 1])
    propagator = Propagator(n_samples= 16 * BIT_WIDTH + 1, window_t=16, central_wl=1)
    shifted_seq_bit = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    pre_shift_output = np.reshape(np.resize(np.repeat(shifted_seq_bit, BIT_WIDTH), propagator.n_samples),
                                (propagator.n_samples, 1))
    evaluator = NormCaseEvaluator(propagator, seq_bit, BIT_WIDTH)
    print('NO NOISE:')
    # print(f'target: {evaluator._target}')
    # print(f'pre_shift_output: {pre_shift_output}')
    # print(f'post_shift_output: {evaluator._align_phase(pre_shift_output)}')
    
    noisy_pre_shift_output = pre_shift_output + np.random.normal(0, 0.05, pre_shift_output.shape[0])
    print('WITH NOISE:')
    print(f'pre_shift_output: {noisy_pre_shift_output}')
    print(f'post_shift_output: {evaluator._align_phase(noisy_pre_shift_output)}')

    assert False # otherwise won't print outputs

