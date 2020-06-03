import autograd.numpy as np
from autograd import grad
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from ..evaluator import Evaluator
from ..assets.functions import power_

"""
Author: Julie Belleville
Start date: June 1, 2020
Resources used: <none>
TODO: refactor into being differentiable under autograd (rn, using filters which isn't)
"""

class MaxEyeEvaluator(Evaluator):
    """
    Given a certain elliptical shape (acting as a mask), find the largest ellipse of that shape
    which will fit in the inside of the eye diagram of the signal (without touching any
    datapoints)

    Meant to be used after an algorithm which evaluates correctness of the pattern,
    as it would by itself give a perfect score to a flat line
    """
# ----------------------------- Public API ------------------------------------

    def __init__(self, propagator, bit_sequence, bit_width, eye_mask=None,
                 mock_graph_for_testing=False, graphical_testing=False):
        """
        Initialises object with an values needed for evaluation

        :param propagator :         propagator object with which evaluation is conducted (propagator)
        :param bit_sequence :       sequence of bits of our target waveform (flattened ndarray with dtype=no.bool)
        :param bit_width :          width of a single sample (in number of propagator) (int)
        :param eye_mask :           shape of the mask with which eye-evaluation and BER will be determined (matplotlib.patches.Polygon)
                                    If None, default eye_mask is used
                                    Note that only the mask shape matters. Size will be scaled in evaluation
        :param mock_graph_for_testing: if true, uses graph passed into the evaluate_graph method as the power
                                       else, uses the graph to get the state and then the power for evaluation
        """
        self.waypoints_num = bit_sequence.shape[0]
        
        if (bit_width * self.waypoints_num > propagator.t.shape[0]):
            raise ValueError("the desired sequence does not fit in the propagator")

        self.propagator = propagator
        self.bit_sequence = bit_sequence
        self._target = np.resize(np.repeat(bit_sequence, bit_width), propagator.n_samples)
        self._bit_width = bit_width 

        self.mock_graph_for_testing = mock_graph_for_testing

        if (eye_mask is None): # use default mask
            self.mask = MaxEyeEvaluator.get_eye_diagram_mask()

        self._graphical_testing = graphical_testing

    @staticmethod
    def get_eye_diagram_mask(semi_major=0.3, semi_minor=0.15):
        """
        Returns a dictionary representing the eye_diagram mask. Mask is an ellipse
        The dictionary contains (key, val) pairs:
            1. ('semi_major', semi_major)
            2. ('semi_minor', semi_minor) (height normalised to 1)
        :param semi_major : semi-major axis length (length is relative to semi-minor, only ratio matters)
        :param semi_minor : semi-minor axis length
        """
        return {
                'semi_major': semi_major,
                'semi_minor': semi_minor,
                }

    def evaluate_graph(self, graph, eval_node=None):
        """
        Returns the inverse of the height of the largest ellipse which can
        fit inside the eye diagram of the data

        :param graph : the graph to evaluate
        :param eval_node : node to evaluate (if None, defaults to the final node)
        """
        if (not self.mock_graph_for_testing):
            if (eval_node is None):
                eval_node = len(graph.nodes) - 1 # set to the last node, assume they've been added in order
            state = graph.nodes[eval_node]['states'][0]
        else:
            state = graph # mock is passed in as graph

        return self._max_eye_inverse(state)

# --------------------------- Helper functions ---------------------------------
    def _max_eye_inverse(self, state):
        """
        Returns the inverse of the max area for our mask shape, such that
        the mask still fits inside the (centered and normalised) eye diagram
        Scaling coefficient scales major and minor axis:: x^2/(a^2) + y^2/(b^2) = s^2 where
        a, b, are the relative major and minor axes, s is the scaling coefficient. Thus, the
        area is A = pi * (a * s) * (b * s) = pi * a * b * s^2

        :param state: the state to test
        :return 1/<max scaling coefficient>
        """
        if (not self.mock_graph_for_testing):
            power = power_(state)
        else:
            power = state
        
        normalized = power / np.max(power)

        # shift datapoints, such that our mask is centred at (0, 0)
        generated = normalized - 0.5 # shift from [0, 1] to [-0.5, 0.5]
        time_tile = np.linspace(-0.5, 0.5, self._bit_width) # normalise time to [-0.5, 0.5).
        time = np.tile(time_tile, (self.bit_sequence.shape[0])) 

        # compute max scaling for each point, and keep only the smallest
        s_squared = (time**2 / self.mask['semi_major']**2 + generated**2 / self.mask['semi_minor']**2).min()
        if (self._graphical_testing):
            self._plot_eye_diagram(time, generated, s_squared)

        return 1 / (np.pi * self.mask['semi_major'] * self.mask['semi_minor'] * s_squared)
    
    def _plot_eye_diagram(self, x, y, scaling):
        """
        Plots the scatterplot x, y of output results (shifted around to (0, 0) on each bit)
        with the maximum sized mask which fits

        :param x : time data
        :param y : power data
        :param scaling : the scaling factor by which to change our mask for it to be at max size
        """
        fig, ax = plt.subplots()
        width = 2 * self.mask['semi_major'] * np.sqrt(scaling)
        height = 2 * self.mask['semi_minor'] * np.sqrt(scaling)

        patch = patches.Ellipse((0, 0), 2 * self.mask['semi_major'] * np.sqrt(scaling),
                                2 * self.mask['semi_minor'] * np.sqrt(scaling),
                                facecolor='orange', alpha=0.3, lw=2)
        ax.add_patch(patch)
        ax.scatter(x, y, color='r')
        plt.title("Max mask (elliptical)")
        plt.show()