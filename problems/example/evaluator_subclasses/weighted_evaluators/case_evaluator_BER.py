import autograd.numpy as np
from autograd import grad

from .weighted_case_evaluator import WeightedEvaluator
from ...assets.functions import power_

"""
Author: Julie Belleville
Start date: May 29, 2020

Resources used: evaluator_rfawg.py (for phase shifting)

TODO: Some possible modifications
1. When we start evaluating more objects, single bit sequence might be replaced by an additional bit_seq parameter
2. Thread-safety? I guess each thread should have its own evaluator, and that might solve everything
3. Address the problem that reshaping doesn't work except if pattern fits perfectly into propagator rn
"""

SIGMOID_EXP = 1000 # kind of arbitrary but should do the trick

class BERCaseEvaluator(WeightedEvaluator):
    """
    This BER based case evaluator will do the following:
    1. Consider the samples in a given bit to determine whether or not a given bit passes
    2. A given bit will pass or fail by its weighted mean being taken (weighting from |sin(wt)|^n)
       Then, bits closer to the centre (where sampling occurs) will be weighed more heavily
    3. Rather than having a strict pass fail, we shall use a sigmoid function for differentiability to attribute
       a score for each bit
    """
# ----------------------------- Public API ------------------------------------

    def __init__(self, propagator, bit_sequence, bit_width, thresh=0.3, weighting_exponent=2,
                 mock_graph_for_testing=False, phase_shift=True):
        """
        Creates comparator object with a set norm and weighting function exponent. 
        Target sequence will be constructed from the bit sequence, and shall have 
        the same length as the propagator arrays. If the propagator array is longer
        than it takes to have the bit sequence once, the target sequence will repeat
        (must be an integer number of times)

        :param propagator : propagator object to use for testing
        :param bit_sequence : target bit sequence (1d np array of booleans)
        :param bit_width : width of a bit, in # of propagator samples
        :param thresh : threshold above 0 or below 1, which qualifies a bit as high or low
                        E.g. if thresh == 0.3, a bit is high if (weighted average of samples) > 0.7
                                               a bit is low if (weighted average of samples) < 0.3
        :param weighting_exponent : exponent n of the weight function |sin(wt)|^n 
        :param mock_graph_for_testing : if true, graph object passed in is treated as the powered
                                        else, graph object is treated as normal

        :raises ValueError if the bit width and sequence length don't fit within the propagator length
        """
        super().__init__(propagator, bit_sequence, bit_width, weighting_exponent,
                         mock_graph_for_testing, phase_shift)
        self.thresh = thresh
        self._bit_width = bit_width
    
    def evaluate_graph(self, graph, propagator, eval_node=None):
        """
        Returns the bit error ratio of the output of a graph when compared to the desired waveform,
        where the bit error ratio = <incorrect bits> / <total bits>.

        Bit correctness is evaluated by the weighted average of the samples within the bit, with more weight
        being given to the central samples.
        
        :param graph : the graph to evaluate
        :param eval_node : if None, last node is evaluated. Else 'eval_node' node is evaluated

        :returns : <incorrect bit number> / <total bit number> (slightly softened to be differentiable)
        """
        if (not self.mock_graph_for_testing):
            if (eval_node is None):
                eval_node = len(graph.nodes) - 1 # set to the last node, assume they've been added in order

            graph.propagate(propagator)
            state = graph.nodes[eval_node]['states'][0]
        else:
            state = graph # mock is passed in as graph

        return self._get_BER(state)

# --------------------------- Helper functions ---------------------------------
    def _get_BER(self, state):
        """
        Returns the BER of a given signal, <sum of all incorrect bits> / <sum of total bits>
        The sum of all incorrect bit is computed as follows:
            1. Take absolute value difference between actual and expected piecewise
            2. Scale these absolute value differences,
               such that centre samples of the propagator weigh more than the outermost samples
            3. Take the average difference over each bit, and feed it into a sigmoid function
               which (with some small differentiability) returns 0 if the error is smaller than self.thresh
               1 otherwise
            4. Sum the result of these sigmoid functions

        :param state : the state to evaluate

        :return : <sum of all incorrect bits> / <sum of total bits>
        """
        if (not self.mock_graph_for_testing):
            power = power_(state)
        else:
            power = state
        
        normalized = power / np.max(power)
        if (self.phase_shift):
            shifted = self._align_phase(normalized)
        else:
            shifted = normalized.reshape((normalized.shape[0], 1))

        # value becomes their weighted difference from target
        weighted_diff = np.abs(self._target - shifted) * self._weighting_funct 
        
        # ------- just for testing -----------
        # norm_val = np.sum(weighted_diff)
        # return norm_val / self.waypoints_num 
        # ------- just for testing -----------

        dist_of_bits = np.sum(
            np.reshape(weighted_diff, (weighted_diff.shape[0] // self._bit_width, self._bit_width)), axis=1)
        
        return np.sum(self._sigmoid(dist_of_bits)) / self.waypoints_num

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * SIGMOID_EXP * (x - self.thresh)))
