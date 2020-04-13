# ! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Parent class for all node-types
"""

import matplotlib.pyplot as plt

from config.config import np

from pint import UnitRegistry
unit = UnitRegistry()

from config.config import np

from .assets.decorators import register_node_types_all
from .assets.functions import fft_, ifft_, psd_, power_

from utils.base_classes import Evaluator as EvaluatorParent

class Evaluator(EvaluatorParent):
    """  Evaluator class, which provides the evaluation of a graph to an objective function.
    """

    __internal_var = None

    def __init__(self):
        super().__init__()



        return

    def evaluate_graph(self, graph):
        """ Function which maps a graph, with set parameters to a score (scalar value) of an objective function """
        return

    def graph_checks(self, graph):
        """ Checks graph to ensure it meets requirements for this evaluation """
        return


