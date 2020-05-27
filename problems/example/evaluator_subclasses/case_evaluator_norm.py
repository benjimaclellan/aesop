import autograd.numpy as np
import timeit

from ..evaluator import Evaluator
from ..assets.functions import power_

"""
Author: Julie Belleville
Start date: May 27, 2020

Resources used: <>

TODO: Some possible modifications
1. When we start evaluating more objects, single bit sequence might be replaced by an additional bit_seq parameter
2. Thread-safety? I guess each thread should have its own evaluator, and that might solve everything
"""

class WeightedNormCaseEvaluator(Evaluator):
    """
    This l-norm based evaluator will do the following:
    1. Take any positive real number norm
    (though tbh, prob will stick with l1 and l2 in practice)
    2. Adjust for global phase shift
    3. Take any non-negative real number power for a sinusoidal weighing function,
    with a period of 2 bits (explained below).
    4. Be differentiable by autograd
    5. Have a runtime comparable to the case_evaluator l-norms


    Sinusoidal weighing function, explained:

    Motivation

    When examining the quality of a bit sequence, the value near the centre of
    the bit is most important: it is what will be mostly likely seen by a detector.
    While an ideal signal would be composed of heaviside functions (i.e. be perfectly square),
    this is impossible with limited bandwidth. Then, it is expected that the bit value does not 
    instantly change on the clock signal: rather, it must be above a certain threshold near the
    centre of the bit.

    It therefore does not make sense to take a pure l-norm, which would punish an inaccurate
    power near the edge of the bit as harshly as in the centre. At the other extreme, looking
    only at the centre value of the bit neglects any fluctuations away from the centre, which is
    also undesirable (after all, a squarer pulse would be ideal, if possible).

    Implementation

    Create a np array with values which form |sin(wt)|^n, for an w = 2pi/T, T = 2 bits of time,
    n is non-negative. The power of n allows us to control the weight of central vs outer samples.
    In the limit n->0, we weigh all points equally. In the limit n->inf, we consider only the central
    values. 
    """
# ----------------------------- Public API ------------------------------------
    def __init__(self, propagator, bit_sequence, bit_width, norm=1, weighting_exponent=2):
        """
        Creates comparator object with a set norm and weighting function exponent

        :param propagator : propagator object to use for testing
        :param bit_sequence : target bit sequence (1d np array of booleans)

        :raises ValueError if the bit width and sequence length aren't consistent with the propagator length
        """
        if (bit_width * bit_sequence.shape[0] != propagator.t.shape[0]):
            raise ValueError("bit width and sequence length are not consistent with the propagator given")

        self.propagator = propagator
        self.bit_sequence = bit_sequence
        self.norm = norm
        self.weighting_funct = self._get_weighting_funct(bit_width, weighting_exponent)
    
    def evaluate_graph(self, graph):
        pass
    
    def graph_checks(self, graph):
        # TODO: figure out whether this is useful for this evaluator
        pass


# --------------------------- Helper functions ---------------------------------

    def _get_weighting_funct(self, bit_width, weighting_exponent):
        """
        Creates the weighting function as described in the class description

        :param bit_width : width of a single bit (in number of samples)
        :param weighting_exponent: exponent to which to raise |sin(wt)|. A higher exponent shifts weight
                                   more towards middle values of the bit
        :returns : an array of the same length as our propagator (the weighting sine wave)
        """
        x = np.arange(self.propagator.t.shape[0], dtype=int)
        w = np.pi / bit_width
        # note: no need to normalise this weighting to an average value of 1,
        # since it will impact all fitness evaluations by the same coefficient 
        # for a given evaluation function
        return np.abs(np.sin(w * x))**weighting_exponent


