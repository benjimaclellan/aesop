import autograd.numpy as np
from autograd import grad

from ..evaluator import Evaluator

"""
Author: Julie Belleville
Start date: June 1, 2020
Resources used: <none>
TODO: refactor into being differentiable under autograd (rn, using filters which isn't)
"""

class MaxEyeEvaluator(Evaluator):
    """
    Given a certain hexagon shape (acting as a mask), find the largest hexagon
    which will fit in the inside of the eye diagram of the signal (without touching any
    datapoints)

    Meant to be used after an algorithm which evaluates correctness of the pattern,
    as it would by itself give a perfect score to a flat line
    """
# ----------------------------- Public API ------------------------------------

    def __init__(self, propagator, bit_sequence, bit_width, eye_mask=None,
                 mock_graph_for_testing=False):
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
            self.mask = MaxEyeEvaluator.get_eye_diagram_mask(bit_width * propagator.dt)

    @staticmethod
    def get_eye_diagram_mask(relative_width=0.5, relative_height=0.3, centre_to_sides_ratio=0.4, centered=False):
        """
        Returns a dictionary representing the eye_diagram mask. Mask is a hexagon with flat top and bottom
        The dictionary contains (key, val) pairs:
            1. ('relative_width', hexagon width / bit width)
            2. ('relative_height', hexagon height) (height normalised to 1)
            3. ('centre_to_Sides_ratio', flat top length / total hexagon width)
        :param relative_width : the total width of the eye diagram
        :param relative_height : the total height of the eye diagram
        :param centre_to_sides_ratio : the ratio of the centre potion (flat top width) to total width
        """
        return {
                'relative_width': width_ratio,
                'relative_height': height_ratio,
                'centre_to_sides_ratio': centre_to_sides_ratio
                }

    def evaluate_graph(self, graph, eval_node=None):
        """
        TODO: docstring
        """
        if (eval_node is None):
            eval_node = len(graph.nodes) - 1 # set to the last node, assume they've been added in order

        if (not self.mock_graph_for_testing):
            state = graph.nodes[eval_node]['states'][0]
        else:
            state = graph # mock is passed in as graph

        return self._max_eye(state)

# --------------------------- Helper functions ---------------------------------
    def _max_eye(self, state):
        """
        TODO: docstring
        """
        if (not self.mock_graph_for_testing):
            power = power_(state)
        else:
            power = state
        
        normalized = power / np.max(power)

        # shift datapoints, such that our mask is centred at (0, 0)
        
        generated = normalized - 0.5 # shift from [0, 1] to [-0.5, 0.5]
        # normalise time to [-0.5, 0.5).
        time_tile = np.linspace(-0.5, 0.5, self._bit_width)
        time = np.tile(time_tile, (self._target.shape[0])) 
            
        # all points with angles in [theta, pi - theta] or [- pi + theta, -theta] from the horizontal restrict the 
        # mask height vertically (that is, h_max < y_max in the range)
        theta = np.arctan2((self.mask['relative_height']),
                        (self.mask['relative_width'] * self.mask['centre_to_sides_ratio']))   
            
        #generate filter for all points which touch the top or bottom of the hexagon
        datapoint_angles = np.arctan2(generated, time)
        filter = np.bitwise_or(np.bitwise_and(theta < datapoint_angles, datapoint_angles < (np.pi - theta)),
                            np.bitwise_and(-np.pi + theta < datapoint_angles, datapoint_angles < -theta))        
        try:
            candidate_max_height = 2 * abs(generated[filter]).min()
        except ValueError: # occurs if no points are in the array, and thus there is no min
            candidate_max_height = 1 # if there are no points in this region, no height bound

        # find max height allowed due to "side" points. Absolute values used to reduce math to first quadrant math, from symmetry
        inverse_filter = np.invert(filter)
        side_y = abs(generated[inverse_filter])
        side_x = abs(time[inverse_filter])
        h = self._mask['relative_height']
        w = self._mask['relative_width']
        m = - h / (w * (1 - self._mask['centre_to_sides_ratio'])) # hexagon side slope
        try:
            candidate_max_height_2 = (2 * h / w * (side_x - side_y / m)).min()
        except ValueError:
            candidate_max_height = 1 

        hex_height = min(candidate_max_height, candidate_max_height_2)
        if (self._graphical_testing):
            mask = CaseEvaluator.get_eye_diagram_mask(1,
                                                    width_ratio=hex_height * w / h,
                                                    height_ratio=hex_height,
                                                    centre_to_sides_ratio=self._mask['centre_to_sides_ratio'],
                                                    centered=True)
                self._plot_eye_diagram(hex_height, filter, mask['path'], time, generated)

            return hex_height
    
    def _plot_eye_diagram(self, hex_height, filter, mask, x, y):
        """
        TODO: docstring
        """
        fig, ax = plt.subplots()
        patch = patches.PathPatch(mask, facecolor='orange', alpha=0.3, lw=2)
        ax.add_patch(patch)
        ax.scatter(x[filter], y[filter], color='r')
        ax.scatter(x[np.invert(filter)], y[np.invert(filter)], color='b')
        plt.title("Max hexagon of height {}".format(hex_height))
        plt.show()