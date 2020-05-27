import autograd.numpy as np
from matplotlib.path import Path
import timeit
# import numpy.polynomial.polynomial as poly
import matplotlib.patches as patches
import matplotlib.pyplot as plt
# import matplotlib.transforms as transforms

from ..evaluator import Evaluator
from ..assets.functions import power_

"""
Author: Julie Belleville
Start date: May 21, 2020
Resources used: 
    https://matplotlib.org/3.2.1/tutorials/advanced/path_tutorial.html

TODO: refactor to allow autograd to do its magic
TODO: add a scaling kernel to weigh the l-norms

"""

class CaseEvaluator(Evaluator):
    """
    The goal of this class is to evaluate given solutions (computational graphs) and
    to attribute them a fitness score. This evaluator's goal is to apply any of the following evaluation methods:
    1. l1-norm
    2. l2-norm
    3. BER (bit error ratio) pure: sum of correct bits (as defined by mask) / total bits
    4. BER scaled: uses elements of the eye (see evaluate_graph for explanation)
    5. eye (max mask): mask of a variable size but fixed shape, with the larger mask. Meant to be used along with one of 1-3
    6. eye (solo): scheme where only the eye diagram evaluation is used (possibly with a negative contribution from incorrect bits?)

    Also evaluated (but this is handled externally, does not directly affect the class):
    1. l1-norm + eye(4 or 5)
    2. l2-norm + eye(4 or 5)
    3. BER + eye(4 or 5)

    where <A> + <B> here signifies a pre-screening of the <B> evaluation with <A> (i.e. evaluate <A>, and then
    add score from <B> evaluation if <A> exceeded a certain threshold). Thus <B> can be used as the real differentiator
    """

# ----------------------------- Public API ------------------------------------
    def __init__(self, propagator, bit_sequence, bit_width, eye_mask=None, thresh_high=0.6, thresh_low=0.4,
                save_runtimes=True, findpoints_runtimes=False, graphical_testing=False):
        """
        Initialises object with an values needed for evaluation

        :param propagator :         propagator object with which evaluation is conducted (propagator)
        :param bit_sequence :       sequence of bits of our target waveform (flattened ndarray with dtype=no.bool)
        :param bit_width :          width of a single sample (in number of propagator) (int)
        :param eye_mask :           shape of the mask with which eye-evaluation and BER will be determined (matplotlib.patches.Polygon)
                                    If None, default eye_mask is used
                                    Note that only the mask shape matters. Size will be scaled in evaluation
        :param thresh_high :        min threshold for a high signal (signal normalised to 1)
        :param thresh_low :         max threshold for a low signal
        :param save_runtimes :      if True, runtimes are saved in self.runtimes (dict with key = function name, val = runtime in s)
        :param findpoints_runtime:  ignored if save_runtimes is false.
                                    If true, saves the time that path.contains_points(datapoints) takes to run


        :raises ValueError : if the number of bits given * the bit_width do not match the propagator length
        """
        # check if bit_sequence * bit_width = propagator length
        if (bit_sequence.shape[0] * bit_width != propagator.t.shape[0]):
            raise ValueError("the expected result array length does not match propagator")

        self.propagator = propagator
        self._bit_width = bit_width
        self._bit_time = bit_width * propagator.dt
        self.target = self._get_target_result(bit_sequence)
        self.target_bit_sequence = bit_sequence
        
        self.thresh_high = thresh_high
        self.thresh_low = thresh_low
        
        if (eye_mask is None):
            self._mask = CaseEvaluator.get_eye_diagram_mask(self._bit_time)
        else:
            self._mask = eye_mask
        
        self.save_runtimes = save_runtimes
        self.find_points_runtime = findpoints_runtimes
        if (save_runtimes):
            self.runtimes = {'l1': None,
                             'l2': None,
                             'BER pure': None,
                             'BER pure alternate': None,
                             'BER with mask': None,
                             'BER scaled': None,
                             'max eye': None,
                             'BER with penalty': None
                            }
            if (findpoints_runtimes):
                self.runtime_breakdown = None # just for testing, so we'll only do it for BER with mask
            self._graphical_testing = graphical_testing

    def evaluate_graph(self, graph, fitness_type, eval_node=None, mocking_graph=False):
        """
        Returns fitness according to one of the following schemes:
            1. l1-norm
            2. l2-norm
            3. BER {pure}: score = <correct bits> / <total bits>. Bits are correct if the average value across the bit pass the threshold
            4. BER {with eye criteria}: score = <correct bits> / <total bits>. A bit is correct if the bit is on the right side, and does not touch the mask 
            4. BER {scaled}: score per  bit is 0 if the bit was incorrect (average of bits above a threshold)
                             score_per_bit = (points outside mask but with correct value (0, 1))/(total points in bit)
                             score = sum of score_per_bit / number_of_bits
                             Basically an eye mask technique
            5. eye {scaled}: score is proportional to the size of the largest mask which fits
            6.BER {with penalty}: Score is computed like in BER scaled, but incorrect bits have a negative contribution
        
        :param graph : the graph to evaluate
        :param fitness_type : 'l1', 'l2', 'BER pure', 'BER with mask', 'BER scaled', 'max eye', 'BER with penalty'
        :param mocking_graph : if False, evaluate graph and take the power as output (as expected)
                               if True, the graph object itself is a (forged) output we're comparing to
        :raises ValueError if fitness_type is not one of the strings above
        """
        if (mocking_graph):
            generated = graph
        else:
            if (eval_node is None):
                eval_node = len(graph.nodes) - 1
            graph.propagate(self.propagator)
            generated = power_(graph.nodes[eval_node]['states'][0])
        
        generated = generated / np.max(generated)  # normalise, because that is extremely necessary
        
        if (fitness_type == 'l1'):
            if (self.save_runtimes):
                start = timeit.default_timer()
                score = self._l_norm_inverse(generated, 1)
                self.runtimes['l1'] = timeit.default_timer() - start
            else:
                score = self._l_norm_inverse(generated, 1)
        elif (fitness_type == 'l2'):
            if (self.save_runtimes):
                start = timeit.default_timer()
                score = self._l_norm_inverse(generated, 2)
                self.runtimes['l2'] = timeit.default_timer() - start
            else:
                score = self._l_norm_inverse(generated, 2)
        elif (fitness_type == 'BER pure'):
            if (self.save_runtimes):
                start = timeit.default_timer()
                score = self._BER_pure(generated)
                self.runtimes['BER pure'] = timeit.default_timer() - start
            else:
                score = self._BER_pure(generated)
        elif (fitness_type == 'BER pure alternate'):
            if (self.save_runtimes):
                start = timeit.default_timer()
                score = self._BER_pure_alternate_implementation(generated)
                self.runtimes['BER pure alternate'] = timeit.default_timer() - start
            else:
                score = self._BER_pure_alternate_implementation(generated)
        elif (fitness_type == 'BER with mask'):
            if (self.save_runtimes):
                start = timeit.default_timer()
                score = self._BER_with_mask(generated)
                self.runtimes['BER with mask'] = timeit.default_timer() - start
            else:
                score = self._BER_with_mask(generated)
        elif (fitness_type == 'BER scaled'):
            if (self.save_runtimes):
                start = timeit.default_timer()
                score = self._BER_scaled(generated)
                self.runtimes['BER scaled'] = timeit.default_timer() - start
            else:
                score = self._BER_scaled(generated)
        elif (fitness_type == 'max eye'):
            if (self.save_runtimes):
                start = timeit.default_timer()
                score = self._max_eye(generated)
                self.runtimes['max eye'] = timeit.default_timer() - start
            else:
                score = self._max_eye(generated)
        elif (fitness_type == 'BER with penalty'):
            if (self.save_runtimes):
                start = timeit.default_timer()
                score = self._BER_scaled(generated, with_penalty=True)
                self.runtimes['BER with penalty'] = timeit.default_timer() - start
            else:
                score = self._BER_scaled(generated, with_penalty=True)
        else:
            raise ValueError("invalid input argument for fitness function type")
        
        return score

    @staticmethod
    def get_eye_diagram_mask(bit_time_interval, width_ratio=0.5, height_ratio=0.3, centre_to_sides_ratio=0.4, centered=False):
        """
        Returns a dictionary representing the eye_diagram mask. Mask is a hexagon with flat top and bottom
        The dictionary contains (key, val) pairs:
            1. ('path', path_object)
            2. ('width_ratio', hexagon width / bit width)
            3. ('height', hexagon height) (height normalised to 1)
            4. ('centre_ratio', flat top length / total hexagon width)

        :param width : the total width of the eye diagram
        :param height : the total height of the eye diagram
        :param centre_ratio : the ratio of the centre potion (flat top width) to total width
        """
        width = width_ratio * bit_time_interval
        if (centered):
            centre_x = 0
            centre_y = 0
        else:
            centre_x = bit_time_interval / 2 # note that this gives a bit of an offset for low bit_widths but won't be a problem in practice
            centre_y = 0.5  # since heights are normalised to 1
        vertices = [
            (centre_x - width / 2, centre_y), 
            (centre_x - width * centre_to_sides_ratio / 2, centre_y + height_ratio / 2),
            (centre_x + width * centre_to_sides_ratio / 2, centre_y + height_ratio / 2),
            (centre_x + width / 2, centre_y),
            (centre_x + width * centre_to_sides_ratio / 2, centre_y - height_ratio / 2),
            (centre_x - width * centre_to_sides_ratio / 2, centre_y - height_ratio / 2),
            (centre_x - width / 2, centre_y), # return to start
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        return {'path': Path(vertices, codes),
                'width_ratio': width_ratio,
                'height_ratio': height_ratio,
                'centre_to_sides_ratio': centre_to_sides_ratio
                }

# ----------------------------- Fitness functions ------------------------------------

    def _l_norm_inverse(self, generated, norm):
        """
        Returns the normalised inverse of the l<norm>-norm between the computational graph output and the target output
        E.g. if norm == 2, returns the (1/{l2-norm})/<number of samples>

        The inverse norm is returned because we want the most similar waveforms to return
        the greatest score.

        :param generated : output generated from the computational graph
        :param norm : integer norm value
        """
        norm_val = np.sum(np.abs(self.target - generated)**norm)**(float(1)/norm)
        if (norm_val < 0.00001): # avoid division by 0
            norm_val = 0.00001
        return (float(1) / norm_val) / self.propagator.n_samples
    
    def _BER_pure(self, generated):
        """
        Returns <correct bits> / <total bits>, where a bit is defined as correct if:
            1. The bit is 1, and <average value across the bit> > thresh_high or
            2. The bit is 0, and <average value across the bit> < thresh_low

        :param generated : output generated from the computational graph
        """
        score = 0
        index = 0
        for bit in np.nditer(self.target_bit_sequence, order='C'):
            avg = np.sum(generated[index * self._bit_width:(index + 1) * self._bit_width]) / self._bit_width
            if (self._bit_value_matches_threshold(bit, avg)):
                # if bit is 1, and central datapoint is high
                score += 1
            index += 1
        return score / self.target_bit_sequence.shape[0]
    
    def _BER_pure_alternate_implementation(self, generated):
        """
        Returns <correct bits> / <total bits>, where a bit is defined as correct if:
            1. The bit is 1, and <average value across the bit> > thresh_high or
            2. The bit is 0, and <average value across the bit> < thresh_low

        :param generated : output generated from the computational graph
        """
        # inspired by: https://stackoverflow.com/questions/4624112/grouping-2d-numpy-array-in-average
        bit_representation = np.mean(generated.reshape((generated.shape[0] // self._bit_width, self._bit_width)), axis=1)
        passes = np.bitwise_or(np.bitwise_and(self.target_bit_sequence, bit_representation > self.thresh_high),
                               np.bitwise_and(np.invert(self.target_bit_sequence), bit_representation < self.thresh_low))
        return np.count_nonzero(passes) / self.target_bit_sequence.shape[0]
    
    def _BER_with_mask(self, generated):
        """
        Returns <correct bits> / <total bits>, where a bit is defined as correct if:
            1. The central value of the bit is on the correct side of its respective threshold and
            2. The points do not fall within the mask

        :param generated : output generated from the computational graph
        """
        score = 0
        index = 0
        self.runtime_breakdown = 0
        for bit in np.nditer(self.target_bit_sequence, order='C'):
            middle = generated[int((index + 0.5) * self._bit_width)]
            if (self._bit_value_matches_threshold(bit, middle) and
                (self._num_violations(bit, index, generated) == 0)): # central value is correct
                score += 1
            index += 1

        return score / self.target_bit_sequence.shape[0]

    def _BER_scaled(self, generated, with_penalty=False):
        """
        Returns sum over all bits of <correctness score> / <total bits>,
        where <correctness score> for a bit is:
            0 if the central bit is incorrect (as per _BER_pure definition)
            1 - <datapoints inside mask> / <datapoints in bit> otherwise
        
        The more a given bit infringes on the mask, the less it contributes to the score

        :param generated : output generated from the computational graph
        """
        # TODO: verify that using only the central bit works well.
        # TODO: consider whether we want the wrong bits to include those on the wrong side
        # (I think not though...? With more points I hope it'll work itself out)
        # Taking the average of the centre 30% of bits or something is also an option
        score = 0
        index = 0
        for bit in np.nditer(self.target_bit_sequence, order='C'):
            middle = generated[int((index + 0.5) * self._bit_width)]
            if (self._bit_value_matches_threshold(bit, middle)): # central value is correct
                score += 1 - (self._num_violations(bit, index, generated) / self._bit_width)
            elif (with_penalty):
                # with penalty penalises wrong result with fewer penalties inside the mask than on the
                # wrong side of the mask
                score -= self._num_violations(bit, index, generated) / self._bit_width
            index += 1

        return score / self.target_bit_sequence.shape[0]
    
    def _max_eye(self, generated):
        """
        Returns <area of the largest mask which fits inside the eye diagram> / <largest mask area for ideal eye diagram>

        Note that the shape of the mask is fixed to self._mask's shape
        :param generated : output generated from the computational graph

        """
        # get datapoints, shifted so that our mask is centred at (0, 0)
        generated = generated - 0.5 # shift from [0, 1] to [-0.5, 0.5]
        # normalise time to [-0.5, 0.5).
        time_tile = np.linspace(-0.5, 0.5, self._bit_width)
        time = np.tile(time_tile, (self.target_bit_sequence.shape[0])) 
        
        # all points with angles in [theta, pi - theta] or [- pi + theta, -theta] from the horizontal restrict the 
        # mask height vertically (that is, h_max < y_max in the range)
        theta = np.arctan2((self._mask['height_ratio']), (self._mask['width_ratio'] * self._mask['centre_to_sides_ratio']))   
        
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
        h = self._mask['height_ratio']
        w = self._mask['width_ratio']
        m = - h / (w * (1 - self._mask['centre_to_sides_ratio'])) # hexagon side slope
        try:
            candidate_max_height_2 = (2 * h / w * (side_x - side_y / m)).min()
        except ValueError:
            candidate_max_height = 1 

        hex_height = min(candidate_max_height, candidate_max_height_2)
        hex_area = self._get_mask_area_from_height(hex_height)
        if (self._graphical_testing):
            mask = CaseEvaluator.get_eye_diagram_mask(1,
                                                            width_ratio=hex_height * w / h,
                                                            height_ratio=hex_height,
                                                            centre_to_sides_ratio=self._mask['centre_to_sides_ratio'],
                                                            centered=True)
            self._plot_eye_diagram(hex_area, filter, mask['path'], time, generated)

        return hex_area
    

# ----------------------------- Additional helpers ------------------------------------

    def _get_target_result(self, bit_sequence):
        """
        Initialises ndarray of target results

        :param bit_sequence : targeted bit sequence
        :param bit_wdidth : width of a single bit
        
        :param generated : output generated from the computational graph
        """
        target_result = np.zeros(bit_sequence.shape[0] * self._bit_width, dtype=int)
        index = 0
        for bit in np.nditer(bit_sequence, order='C'):
            if bit: # if the bit should be one, set the corresponding parts of the target signal to 1
                target_result[index * self._bit_width:(index + 1) * self._bit_width] = 1
            index += 1
    
        return target_result

    def _num_violations(self, bit, bit_index, generated, mask=None):
        # grab relevant time slice, but shift all so that first value is 0
        if (mask is None):
            mask = self._mask['path']
        time = (self.propagator.t[bit_index * self._bit_width: (bit_index + 1) * self._bit_width] - \
               self.propagator.t[bit_index * self._bit_width]).flatten()
        new_generated = generated[bit_index * self._bit_width: (bit_index + 1) * self._bit_width]
        
        datapoints = np.stack((time, new_generated), axis=-1)

        if (self.find_points_runtime):
            start = timeit.default_timer()
        inside_mask_filter = mask.contains_points(datapoints)
        num_in_mask = np.count_nonzero(inside_mask_filter)

        if (self.find_points_runtime):
            self.runtime_breakdown += timeit.default_timer() - start

        wrong_side_filter = new_generated < 0.5 if bit else new_generated > 0.5
        mask_time_filter = np.bitwise_or(self._bit_time * (0.5 - self._mask['width_ratio']/2) < time, time < self._bit_time * (0.5 + self._mask['width_ratio']/2))

        incorrect_filter  = np.bitwise_and(np.invert(inside_mask_filter), np.bitwise_and(wrong_side_filter, mask_time_filter))
                            
        num_in_mask += np.count_nonzero(incorrect_filter)

        if (self._graphical_testing):
            self._plot_mask_points(num_in_mask, time, new_generated)

        return num_in_mask

    def _bit_value_matches_threshold(self, bit, val):
        """
        Returns True if the bit is high and the value is above the high threshold or
                     if the bit is low and the value is below the low threshold
                False otherwise
        :param bit : the bit which we're checking
        :param val : the value
        """
        return (bit and (val > self.thresh_high)) or ((not bit) and (val < self.thresh_low))
    
    def _get_mask_area_from_height(self, height):
        """
        Returns area of the mask shaped like self._mask with height as given

        :param height : the height of the mask
        :return : area
        """
        width = height * self._mask['width_ratio'] / self._mask['height_ratio']
        return 0.5 * height * (1 + width * self._mask['centre_to_sides_ratio'])
    
    def _plot_mask_points(self, num_in_mask, time, power):
        mask = self._mask['path'] # .translate(-self._bit_width / 2, 0)
        fig, ax = plt.subplots()
        patch = patches.PathPatch(mask, facecolor='orange', alpha=0.3, lw=2)
        ax.add_patch(patch)
        ax.scatter(time, power, color='r')
        plt.title("Number of invalid datapoints (in mask or wrong side): {}".format(num_in_mask))
        plt.show()

    def _plot_eye_diagram(self, hexagon_area, filter, mask, x, y):
        fig, ax = plt.subplots()
        patch = patches.PathPatch(mask, facecolor='orange', alpha=0.3, lw=2)
        ax.add_patch(patch)
        ax.scatter(x[filter], y[filter], color='r')
        ax.scatter(x[np.invert(filter)], y[np.invert(filter)], color='b')
        plt.title("Max hexagon of area {}".format(hexagon_area))
        plt.show()

# ----------------------------- Test functions ------------------------------------

# transferred to test_case_evaluator_bin.py