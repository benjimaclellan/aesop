"""

"""

# this sets which numpy library to use throughout - either numpy or autograd.numpy
import numpy as np
import scipy as sp


# TODO: dynamically choose which simulation library to use

# set plotting style
# import matplotlib.pyplot as plt
# plt.style.use(r"config/plot-style.mplstyle")

# plug-in dictionaries to collect available functions/classes
EVOLUTION_OPERATORS = dict()
NODE_TYPES = dict()
NODE_TYPES_ALL = dict()