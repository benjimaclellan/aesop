import pytest
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math

from problems.example.evaluator_subclasses.case_evaluator_bin import CaseEvaluatorBinary
from problems.example.assets.propagator import Propagator

# TODO:
# 1. Setup 1 random series of bits (pseudorandom) of length 15, use bit_width = 3
# 2. Setup 1 waveform that is an exact polynomial approximation through those points
#          1 waveform that is the exact function we want
#          1 waveform that is the reverse of the exact function we wan
#          1 polyomial approximation with a few bit flips
#          1 polyomial approximation with another pseudo-random sequence (should be about 50% wrong)
#          1 fourier approximation with noise thrown in (maybe a few bit flips) ?

TESTING_WITH_PLOT = True

@pytest.fixture(scope='module')
def rand_bit_seq():
    # 00010101
    return np.array([0, 0, 0, 1, 0, 1, 0, 1], dtype=bool)


@pytest.fixture(scope='module')
def graph_mockup_exact_poly(time_arr):
    x = np.array([1, 4, 7, 10, 13, 16, 19, 22])
    y = np.array([0, 0, 0, 1, 0, 1, 0, 1])
    coefs = poly.polyfit(x, y, 7)

    ffit = poly.polyval(time_arr, coefs)
    return abs(ffit)


@pytest.fixture(scope='module')
def graph_mockup_exact_square():
    return np.array([0, 0, 0,
                     0, 0, 0,
                     0, 0, 0,
                     1, 1, 1,
                     0, 0, 0, 
                     1, 1, 1,
                     0, 0, 0,
                     1, 1, 1])


@pytest.fixture(scope='module')
def graph_mockup_reverse_square():
    return np.array([1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     0, 0, 0,
                     1, 1, 1, 
                     0, 0, 0,
                     1, 1, 1,
                     0, 0, 0])

@pytest.fixture(scope='module')
def graph_mockup_approx():
    return np.array([0.3, 0.15, 0.1,
                     0.2, 0.12, 0.2,
                     0.2, 0.3, 0.5,
                     0.8, 1, 0.4,
                     0.3, 0.2, 0.4, 
                     0.7, 0.95, 0.75,
                     0.35, 0.1, 0.6,
                     0.8, 0.9, 1])

@pytest.fixture(scope='module')
def graph_mockup_bitflip():
    return np.array([0.3, 0.15, 0.1,
                     0.65, 0.9, 0.85,
                     0.2, 0.3, 0.5,
                     0.8, 1, 0.4,
                     0.3, 0.2, 0.4, 
                     0.7, 0.95, 0.75,
                     0.35, 0.1, 0.6,
                     0.8, 0.9, 1])


@pytest.fixture(scope='module')
def time_arr():
    return np.arange(0, 24, 1)


@pytest.fixture(scope='module')
def basic_evaluator(rand_bit_seq):
    #TODO: add default value to central_wl so that the code doesn't fail when it's not specified
    propagator = Propagator(n_samples=24, window_t=25, central_wl=1)
    return CaseEvaluatorBinary(propagator, rand_bit_seq, 3, graphical_testing=TESTING_WITH_PLOT)

def test_mask_creation():
    """
    Tests mask creation, and confirms that mask['path].contains_points
    will correctly identify points inside the mask
    """
    mask_full = CaseEvaluatorBinary.get_eye_diagram_mask(1)
    mask = mask_full['path']
    fig, ax = plt.subplots()
    patch = patches.PathPatch(mask, facecolor='orange', alpha=0.3, lw=2)
    ax.add_patch(patch)
    x_coords = np.array([0, 0.2, 0.7, 0.5, 0.37, 0.8, 0.25])
    y_coords = np.array([0, 0.3, 0.6, 0.5, 0.45, 0.3, 0.40])
    points = np.stack((x_coords, y_coords), axis=-1)
    contains = mask.contains_points(points)
    
    if (TESTING_WITH_PLOT):
        print(points)
        ax.scatter(x_coords, y_coords, color='r')
        plt.show()
        print(contains)
    assert contains[0] == False
    assert contains[1] == False
    assert contains[2] == False
    assert contains[3] == True
    assert contains[4] == True
    assert contains[5] == False
    assert contains[6] == False


def test_get_target_result(graph_mockup_exact_square, rand_bit_seq, basic_evaluator):
    assert np.array_equal(basic_evaluator._get_target_result(rand_bit_seq), graph_mockup_exact_square)


def test_1_norm_poly(graph_mockup_exact_poly, graph_mockup_exact_square, basic_evaluator):
    graph = graph_mockup_exact_poly / np.max(graph_mockup_exact_poly)
    diff = np.sum(np.abs(graph - graph_mockup_exact_square))
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_exact_poly, 'l1', mocking_graph=True), (1 / diff) / 24)


def test_1_norm_exact_square(graph_mockup_exact_square, basic_evaluator):
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_exact_square, 'l1', mocking_graph=True), 100000 / 24)


def test_1_norm_reverse_square(graph_mockup_reverse_square, basic_evaluator):
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_reverse_square, 'l1', mocking_graph=True), (1 / 24) / 24)


def test_1_norm_approx(graph_mockup_approx, basic_evaluator):
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_approx, 'l1', mocking_graph=True), (1 / 5.72) / 24)


def test_1_norm_bitflip(graph_mockup_bitflip, basic_evaluator):
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_bitflip, 'l1', mocking_graph=True), (1 / 7.6) / 24)


def test_2_norm_approx(basic_evaluator, graph_mockup_approx):
    # norm probably just requires one test let's be real
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_approx, 'l2', mocking_graph=True), (1 / math.sqrt(1.9844)) / 24)


def test_BER_pure_exact_square(graph_mockup_exact_square, basic_evaluator):
    assert basic_evaluator.evaluate_graph(graph_mockup_exact_square, 'BER pure', mocking_graph=True) == 1
    

def test_BER_pure_reverse_square(graph_mockup_reverse_square, basic_evaluator):
    assert basic_evaluator.evaluate_graph(graph_mockup_reverse_square, 'BER pure', mocking_graph=True) == 0


def test_BER_pure_approx(graph_mockup_approx, basic_evaluator):
    assert basic_evaluator.evaluate_graph(graph_mockup_approx, 'BER pure', mocking_graph=True) == 1


def test_BER_pure_bitflip(graph_mockup_bitflip, basic_evaluator):
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_bitflip, 'BER pure', mocking_graph=True), 7 / 8)


def test_BER_mask_approx(graph_mockup_approx, basic_evaluator):
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_approx, 'BER with mask', mocking_graph=True), 7 / 8)
    

def test_BER_mask_bitflip(graph_mockup_bitflip, basic_evaluator):
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_bitflip, 'BER with mask', mocking_graph=True), 6 / 8)


def test_BER_scaled_approx(graph_mockup_approx, basic_evaluator):
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_approx, 'BER scaled', mocking_graph=True), 7 / 8)


def test_BER_scaled_bitflip(graph_mockup_bitflip, basic_evaluator):
    assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_bitflip, 'BER scaled', mocking_graph=True), 6 / 8)


def test_BER_scaled_with_penalty_approx(graph_mockup_approx, basic_evaluator):
     assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_approx, 'BER with penalty', mocking_graph=True), 7 / 8)


def test_BER_scaled_with_penalty_bitflip(graph_mockup_bitflip, basic_evaluator):
     assert math.isclose(basic_evaluator.evaluate_graph(graph_mockup_bitflip, 'BER with penalty', mocking_graph=True), 5 / 8)


def test_max_eye():
    pass