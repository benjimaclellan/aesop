import autograd.numpy as np
import pytest
from autograd import grad


# these two functions should be functionally the same but let's cover those bases
def basic_gaussian_normal(x):
    return np.random.normal(loc=x[0], scale=x[1])

def basic_gaussian_randn(x):
    return np.random.randn() * x[1] + x[0]

def test_basic_gaussian_normal():
    x = np.array([0, 1])

    grad_normal = grad(basic_gaussian_normal)
    grad_randn = grad(basic_gaussian_normal)

    print(grad_normal(x))
    print(grad_randn(x))