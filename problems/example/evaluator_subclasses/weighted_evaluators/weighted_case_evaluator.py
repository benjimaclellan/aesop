import autograd.numpy as np
from autograd import grad

from ...evaluator import Evaluator

"""
Author: Julie Belleville
Start date: May 29, 2020

Parent class for all weighted evaluators
"""

class WeightedEvaluator(Evaluator):
    """
    Parent class for all weighted evaluators, providing a weighting funct providing function, 
    and phase shift alignment funct

    
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

    def __init__(self, propagator, bit_sequence, bit_width, weighting_exponent,
                 mock_graph_for_testing, phase_shift):
        """
        Creates comparator object with a set norm and weighting function exponent. 
        Target sequence will be constructed from the bit sequence, and shall have 
        the same length as the propagator arrays. If the propagator array is longer
        than it takes to have the bit sequence once, the target sequence will repeat
        (must be an integer number of times)

        :param propagator : propagator object to use for testing
        :param bit_sequence : target bit sequence (1d np array of booleans)

        :raises ValueError if the bit width and sequence length don't fit within the propagator length
        """
        self.waypoints_num = bit_sequence.shape[0]
    
        if (bit_width * self.waypoints_num > propagator.t.shape[0]):
            raise ValueError("the desired sequence does not fit in the propagator")
        
        self.propagator = propagator
        self._bit_width = bit_width
        self._bit_sequence = bit_sequence
        self._weighting_exp = weighting_exponent
        self._weighting_funct = self._get_weighting_funct(bit_width, weighting_exponent)

        self._target = np.reshape(np.resize(np.repeat(bit_sequence, bit_width), propagator.n_samples),
                                  (propagator.n_samples, 1))

        # for adjusting phase difference
        self.phase_shift = phase_shift
        if (phase_shift):
            self._target_harmonic = 1 / (bit_width * self.waypoints_num * self.propagator.dt)
            self._target_rf = np.fft.fft(self._target, axis=0)

            self._phase_shift_arr = (np.fft.fftshift(
                np.linspace(0, len(self._target_rf) - 1, len(self._target_rf))) / self.propagator.n_samples
            ).reshape((self.propagator.n_samples, 1)) # make column vector

        # if true, graph evaluation is replaced by a passed in state (for testing)
        self.mock_graph_for_testing = mock_graph_for_testing
    
    @property
    def weighting_exp(self):
        return self._weighting_exp
    
    @weighting_exp.setter
    def weighting_exp(self, exp):
        self._weighting_funct = self._get_weighting_funct(self.bit_width, exp)
        self._weighting_exp = exp
    
    @property
    def bit_sequence(self):
        return self._bit_sequence
    
    @bit_sequence.setter
    def bit_sequence(self, bit_seq):
        self._target = np.reshape(np.resize(np.repeat(bit_seq, self.bit_width), self.propagator.n_samples),
                                  (self.propagator.n_samples, 1))
        self._bit_sequence = bit_seq

# --------------------------- Helper functions ---------------------------------

    def _get_weighting_funct(self, bit_width, weighting_exponent):
        """
        Creates the weighting function as described in the class description

        :param bit_width : width of a single bit (in number of samples)
        :param weighting_exponent: exponent to which to raise |sin(wt)|. A higher exponent shifts weight
                                   more towards middle values of the bit
        :returns : an array of the same length as our propagator (the weighting sine wave)
        """
        x = np.arange(self.propagator.n_samples, dtype=int)
        w = np.pi / bit_width
        
        waveform = np.abs(np.sin(w * (x + 0.5)))**weighting_exponent
        normalising_factor = np.sum(waveform[0:bit_width]) # normalising is necessary for BER evaluation for example

        return np.reshape(waveform, (waveform.shape[0], 1)) / normalising_factor

    def _align_phase(self, state_power):
        """
        Basically adapted from shift_function in evaluator_rfawg.py

        TODO: validate effectiveness
        """
        state_rf = np.fft.fft(state_power, axis=0).reshape((state_power.shape[0], 1))
        target_harmonic_index = (np.rint(self._target_harmonic / self.propagator.df)).astype('int')
        # target_harmonic_index = (self._target_harmonic / self.propagator.df).astype('int')    

        phase = np.angle(state_rf[target_harmonic_index] / self._target_rf[target_harmonic_index])
        shift = phase / (self._target_harmonic * self.propagator.dt)

        state_rf *= np.exp(-1j * shift * self._phase_shift_arr)

        return np.abs(np.fft.ifft(state_rf, axis=0))
