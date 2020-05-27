import pytest
import autograd.numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math

from problems.example.evaluator_subclasses.case_evaluator_norm import WeightedNormCaseEvaluator as CaseEvaluator
from problems.example.assets.propagator import Propagator


GRAPHICAL_TESTING = True
BASIC_BIT_WIDTH = 4

@pytest.fixture(scope='module')
def bit_seq_to_array_converter():
    def target_seq(bit_width, bit_seq):
        """
        Initialises ndarray of target results. 
        WARNING: Code is inefficient, which really
        could be fixed, but this method should only really be used
        by self.plot_target_vs_weighting for some super limited test case.
            
        :param generated : output generated from the computational graph
        """
        target_result = np.zeros(bit_seq.shape[0] * bit_width, dtype=int)
        index = 0
        for bit in bit_seq:
            if bit: # if the bit should be one, set the corresponding parts of the target signal to 1
                target_result[index * bit_width:(index + 1) * bit_width] = 1
            index += 1
        
        return target_result
    return target_seq


@pytest.fixture(scope='module')
def bit_seq_4_bytes():
    return np.array([0, 1, 0, 1, 0, 1, 0, 1,
                     0, 1, 0, 1, 0, 1, 0, 1,
                     0, 1, 0, 1, 0, 1, 0, 1,
                     0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)


@pytest.fixture(scope='module')
def propagator():
    return Propagator(n_samples=128, window_t=1, central_wl=1)
    

@pytest.fixture(scope='module')
def norm1_exp2_evaluator(bit_seq_4_bytes, propagator):
    return CaseEvaluator(propagator, bit_seq_4_bytes, BASIC_BIT_WIDTH)


@pytest.mark.skipif(not GRAPHICAL_TESTING, reason='no graphical testing active')
def test_target_vs_weighting_plot(bit_seq_4_bytes, norm1_exp2_evaluator, bit_seq_to_array_converter):
    target_array = bit_seq_to_array_converter(BASIC_BIT_WIDTH, bit_seq_4_bytes)

    fig, ax = plt.subplots()
    x = np.arange(norm1_exp2_evaluator.propagator.t.shape[0], dtype=int)

    ax.plot(x, target_array, color='r', label='target')
    ax.plot(x, norm1_exp2_evaluator.weighting_funct, color='b', label='weighting function')
    plt.show()