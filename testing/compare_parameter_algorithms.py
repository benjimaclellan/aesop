import warnings

# %% import public modules
import matplotlib.pyplot as plt
import autograd.numpy as np
import numpy
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

# %% import custom modules
from lib.analysis.hessian import hessian
from testing.test_functions import TestFunction
import cma

# booth_function, matyas_function, easom_function, ackley_function
# rastrigin_function, sphere_function, rosenbrock_function, beale_function
# goldstein_price_function, levi13_function, eggholder_function,gaussian_function

# %%
"""
All functions defined from:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

plt.close("all")

if __name__ == '__main__':


    plot_types = set(('3d', '2d'))

    test = TestFunction('beale_function')
    print(test.xP)

    res = cma.fmin(test.function,
                   [(test.x_range[1]-test.x_range[0])/2, (test.x_range[1]-test.x_range[0])/2],
                   1)
