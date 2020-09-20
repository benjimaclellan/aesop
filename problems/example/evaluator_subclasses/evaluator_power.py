""" """

import autograd.numpy as np
from ..evaluator import Evaluator

class PeakPower(Evaluator):
    """  """

    def __init__(self, propagator, **kwargs):
        super().__init__(**kwargs)


    def evaluate_graph(self, graph, propagator):
        evaluation_node = [node for node in graph.nodes if not graph.out_edges(node)][0]  # finds node with no outgoing edges
        graph.propagate(propagator)
        state = graph.measure_propagator(evaluation_node)
        score = np.max(np.power(np.abs(state), 2))
        return score


