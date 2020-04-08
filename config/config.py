#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

# this sets which numpy library to use throughout - either numpy or autograd.numpy
import numpy as np

# TODO: dynamically choose which simulation library to use

# set plotting style
# import matplotlib.pyplot as plt
# plt.style.use(r"config/plot-style.mplstyle")

# plug-in dictionaries to collect available functions/classes
EVOLUTION_OPERATORS = dict()
NODE_TYPES = dict()
NODE_TYPES_ALL = dict()