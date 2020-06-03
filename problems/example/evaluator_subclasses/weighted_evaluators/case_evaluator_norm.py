import autograd.numpy as np
from autograd import grad

from .weighted_case_evaluator import WeightedEvaluator
from ...assets.functions import power_

"""
Author: Julie Belleville
Start date: May 27, 2020

Resources used: evaluator_rfawg.py (for phase shifting)

TODO: Some possible modifications
1. When we start evaluating more objects, single bit sequence might be replaced by an additional bit_seq parameter
2. Thread-safety? I guess each thread should have its own evaluator, and that might solve everything
"""

class NormCaseEvaluator(WeightedEvaluator):
    """
    This l-norm based evaluator will do the following:
    1. Take any positive real number norm
    (though tbh, prob will stick with l1 and l2 in practice)
    2. Adjust for global phase shift
    3. Take any non-negative real number power for a sinusoidal weighing function,
    with a period of 2 bits (explained below).
    4. Be differentiable by autograd
    5. Have a runtime comparable to the case_evaluator l-norms
    """
# ----------------------------- Public API ------------------------------------
    def __init__(self, propagator, bit_sequence, bit_width, norm=1, weighting_exponent=2,
                 mock_graph_for_testing=False):
        """
        Creates comparator object with a set norm and weighting function exponent. 
        Target sequence will be constructed from the bit sequence, and shall have 
        the same length as the propagator arrays. If the propagator array is longer
        than it takes to have the bit sequence once, the target sequence will repeat
        (must be an integer number of times)

        :param propagator : propagator object to use for testing
        :param bit_sequence : target bit sequence (1d np array of booleans)
        :param bit_width : width of a bit, in # of propagator samples
        :param norm : norm float to use for the evaluate
                      E.g. if norm == 1, use l1-norm 
        :param weighting_exponent : exponent n of the weight function |sin(wt)|^n 
        :param mock_graph_for_testing : if true, graph object passed in is treated as the powered
                                        else, graph object is treated as normal

        :raises ValueError if the bit width and sequence length don't fit within the propagator length
        """
        super().__init__(propagator, bit_sequence, bit_width, weighting_exponent, mock_graph_for_testing)
        self.norm = norm
    
    def evaluate_graph(self, graph, eval_node=None):
        """
        Returns the self.norm-norm between our desired target waveform, and the normalized generated waveform,
        weighted by our (sinusoidal to a power) weighting function.
        If self.mock_graph_for_testing, graph is the power array (testing purposes only). Else, it's the actual graph

        :param graph : the graph to evaluate
        :param eval_node : if None, last node is evaluated. Else 'eval_node' node is evaluated

        :returns: (sum of |target - normalized|**norm)**(1/norm) * |sin(wt)|**n / number of samples
        """
        if (not self.mock_graph_for_testing):
            if (eval_node is None):
                eval_node = len(graph.nodes) - 1 # set to the last node, assume they've been added in order
            
            state = graph.nodes[eval_node]['states'][0]
        else:
            state = graph # mock is passed in as graph

        return self._get_norm(state)

    def graph_checks(self, graph):
        # TODO: figure out whether this is useful for this evaluator
        pass

# --------------------------- Helper functions ---------------------------------
    
    def _get_norm(self, state):
        """
        Gets the norm as described in evaluate_graph
        :param state : the state output by the graph (or the power itself, if self.mock_graph_for_testing)
        """
        if (not self.mock_graph_for_testing):
            power = power_(state)
        else:
            power = state
        
        normalized = power / np.max(power)
        shifted = self._align_phase(normalized)
        weighted_diff = np.abs(self._target - shifted) * self.weighting_funct
        norm_val = np.sum(weighted_diff**self.norm)**(float(1)/self.norm)

        return norm_val / self.waypoints_num