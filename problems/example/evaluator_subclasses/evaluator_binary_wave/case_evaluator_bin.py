import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from ...evaluator import Evaluator
from ..assets.functions import power_

"""
Author: Julie Belleville
Start date: May 21, 2020
Resources used: 
    https://matplotlib.org/3.2.1/tutorials/advanced/path_tutorial.html
"""

class CaseEvaluatorBinary(Evaluator):
    """
    The goal of this class is to evaluate given solutions (computational graphs) and
    to attribute them a fitness score. This evaluator's goal is to apply any of the following evaluation methods:
    1. l1-norm
    2. l2-norm
    3. BER (bit error ratio)
    4. eye (set mask): mask of a fixed size, with penalties for violations of the mask. Meant to be used along with one of 1-3
    5. eye (max mask): mask of a variable size but fixed shape, with the larger mask. Meant to be used along with one of 1-3
    6. eye (solo): scheme where only the eye diagram evaluation is used (possibly with a negative contribution from incorrect bits?)

    Also evaluated (but this is handled externally, does not directly affect the class):
    1. l1-norm + eye(4 or 5)
    2. l2-norm + eye(4 or 5)
    3. BER + eye(4 or 5)

    where <A> + <B> here signifies a pre-screening of the <B> evaluation with <A> (i.e. evaluate <A>, and then
    add score from <B> evaluation if <A> exceeded a certain threshold). Thus <B> can be used as the real differentiator
    """

    def __init__(propagator, bit_sequence, bit_width, eye_mask=None, thresh_high=0.75, thresh_low=0.25):
        """
        Initialises object with an values needed for evaluation

        :param propagator :     propagator object with which evaluation is conducted (propagator)
        :param bit_sequence :   sequence of bits of our target waveform (flattened ndarray with dtype=no.bool)
        :param bit_width :      width of a single sample (in number of propagator) (int)
        :param eye_mask :       shape of the mask with which eye-evaluation and BER will be determined (matplotlib.patches.Polygon)
                                If None, default eye_mask is used
                                Note that only the mask shape matters. Size will be scaled in evaluation
        :param thresh_high :    min threshold for a high signal (signal normalised to 1)
        :param thresh_low :     max threshold for a low signal

        :raises ValueError : if the number of bits given * the bit_width do not match the propagator length
        """
        # check if bit_sequence * bit_width = propagator length
        if (bit_sequence.shape[0] * bit_width != propagator.shape[0]):
            raise ValueError("the expected result array length does not match propagator")

        self.propagator = propagator
        self._bit_time = bit_width * propagator.dt
        self.target = self._get_target_result(bit_sequence)
        
        self.thresh_high = thresh_high
        self.thresh_low = thresh_low
        
        if (eye_mask is None):
            self._mask = cls.get_eye_diagram_mask(self._bit_time)
        else:
            self._mask = mask

    def evaluate_graph(graph, fitness_type, eval_node=None):
        """
        Returns fitness according to one of the following schemes:
            1. l1-norm
            2. l2-norm
            3. BER (bit error ratio)
            4. l1-norm + eye
            5. l2-norm + eye
            6. BER + eye
        
        :param graph : the graph to evaluate
        :param fitness_type : 'l1', 'l2', 'BER', 'set eye', 'max eye', 'solo eye'
        :raises ValueError if fitness_type is not one of the strings above
        """
        if (eval_node is None):
            eval_node = len(graph.nodes) - 1

        graph.propagate(self.propagator)
        generated = power_(graph.nodes[eval_node]['states'][0])
        generated = generated / np.max(generated)  # normalise, because that is extremely necessary
        
        if (fitness_type == 'l1'):
            return self._l_norm_inverse(generated, 1)
        elif (fitness_type == 'l2'):
            return self._l_norm_inverse(generated, 2)
        elif (fitness_type == 'BER'):
            pass
        elif (fitness_type == 'set eye'):
            pass
        elif (fitness_type == 'max eye'):
            pass
        elif (fitness_type == 'solo eye'):
            pass
        else:
            raise ValueError("invalid input argument for fitness function type")

    @staticmethod
    def get_eye_diagram_mask(bit_time_interval, width=0.3, height=0.3, centre_ratio=0.4):
        """
        Returns a path representing the eye_diagram mask. Mask is a hexagon with flat top and bottom

        :param width : the total width of the eye diagram
        :param height : the total height of the eye diagram
        :param centre_ratio : the ratio of the centre potion (flat top width) to total width
        """
        centre_x = bit_time_interval / 2
        centre_y = 0.5  # since heights are normalised to 1
        vertices = [
            (centre_x - width / 2, centre_y), 
            (centre_x - width * centre_ratio / 2, centre_y + height / 2),
            (centre_x + width * centre_ratio / 2, centre_y + height / 2),
            (centre_x + width / 2, centre_y),
            (centre_x + width * centre_ratio / 2, centre_y - height / 2),
            (centre_x - width * centre_ratio / 2, centre_y - height / 2),
            (centre_x - width / 2, centre_y), # return to start
        ]

        codes = [
            Path.MOVETO,
            Path.LINE_TO,
            Path.LINE_TO,
            Path.LINE_TO,
            Path.LINE_TO,
            Path.LINE_TO,
            Path.CLOSEPOLY,
        ]

        return Path(vertices, codes)

    
    def _get_target_result(self, bit_sequence):
        """
        Initialises ndarray of target results

        :param bit_sequence : targeted bit sequence
        :param bit_wdidth : width of a single bit
        """
        target_result = np.zeros(bit_sequence.shape[0] * bit_width, dtype=int)
        for bit in np.nditer(bit_sequence, order='C'):
            if bit: # if the bit should be one, set the corresponding parts of the target signal to 1
                target_result[bit.index * self._bit_width:(bit.index + 1) * self._bit_width] = 1

        return target_result

    def _l_norm_inverse(self, generated, norm):
        """
        Returns the normalised inverse of the l<norm>-norm between the computational graph output and the target output
        E.g. if norm == 2, returns the (1/{l2-norm})/<number of samples>

        The inverse norm is returned because we want the most similar waveforms to return
        the greatest score.

        :param state : output state from the computational graph
        :param norm : integer norm value
        """
        norm_val = np.sum(np.abs(self.target - generated)**norm)**(float(1)/norm)
        return (float(1) / norm_val) / self.propagator.n_samples

def test_poly():
    mask = CaseEvaluatorBinary.get_eye_diagram_mask(1)
    fig, ax = plt.subplots()
    patch = patches.PathPatch(mask, facecolor='orange', lw=2)
    ax.add_patch(patch)
    plt.show()

if __name__ == "__main__":
    test_poly()