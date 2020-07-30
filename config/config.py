"""

"""

# this sets which numpy library to use throughout - either numpy or autograd.numpy
import autograd.numpy as np   # TODO: maybe just always use autograds wrapping of numpy
import scipy as sp

import matplotlib.pyplot as plt
import os
# set plotting style

plt.style.use(os.path.join(os.path.dirname(__file__), r"plot-style.mplstyle"))

# TODO: dynamically choose which simulation library to use

# plug-in dictionaries to collect available functions/classes
EVOLUTION_OPERATORS = dict()
NODE_TYPES = dict()
NODE_TYPES_ALL = dict()

LOG_DIRECTORY = r"C:\Users\benjamin\Documents\INRS - Projects\asope\logs"