import pytest
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from problems.example.evaluator_subclasses.case_evaluator_bin import CaseEvaluatorBinary

# TODO:
# 1. Setup 1 random series of bits (pseudorandom) of length 10, use bit_width = 5
# 2. Setup 1 waveform that is an exact polynomial approximation through those points
#          1 waveform that is the exact function we want
#          1 waveform that is the reverse of the exact function we wan
#          1 polyomial approximation with a few bit flips
#          1 polyomial approximation with another pseudo-random sequence (should be about 50% wrong)
#          1 fourier approximation with noise thrown in (maybe a few bit flips) ?

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
    print(points)
    ax.scatter(x_coords, y_coords, color='r')
    plt.show()
    contains = mask.contains_points(points)
    print(contains)
    assert contains[0] == False
    assert contains[1] == False
    assert contains[2] == False
    assert contains[3] == True
    assert contains[4] == True
    assert contains[5] == False
    assert contains[6] == False

def test_get_target_result():
    pass

def test_1_norm():
    pass

def test_2_norm():
    pass

def test_BER_pure():
    pass

def test_BER_mask():
    pass

def test_BER_scaled():
    pass

def test_BER_scaled_with_penalty():
    pass

def test_max_eye():
    pass