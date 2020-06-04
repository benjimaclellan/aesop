from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm as tuning_GA
from problems.example.evaluator_subclasses.weighted_evaluators.case_evaluator_norm import NormCaseEvaluator
from problems.example.evaluator_subclasses.weighted_evaluators.case_evaluator_BER import BERCaseEvaluator

"""
The goal of the script is to gather data about the relative performances of norm and BER evaluation methods (as
well as mixed method involving both norm/BER and the max_eye evaluator) with different norm and weighting coefficients
(see NormCaseEvaluator and BERCaseEvaluator for further details on the meaning of these parameters).

Evaluation procedure:
1. For each bit sequence length, generate 3 random (but reproducible) sequences
2. On each of these sequences, 

TODO: how to represent the structure which holds all my logs?
TODO: once topological GA is set up, remove the hardcoded topology and run the full GA
TODO: add max eye metric into the mix (implement multi-tiered evaluation, which will mess up the log metrics...)
TODO: verify that all metrics are normalised to 1, and if they're not, do it
TODO: figure out how to incorporate stability of solution
TODO: add visualisation based on log data
"""

# define testing parameters
SEQUENCE_LENGTHS = [8, 32, 128]
L_NORM_VALS = [1, 2]
WEIGHTING_COEFS = [0, 2, 4, 8, 16]
NUM_SEQS_OF_FIXED_LENGTH = 3 # how many sequencse of the same length to test

# build a dictionary structure to hold
log_dict = {} # 1st key in SEQUENCE_LENGTH, 2nd key weighting coeffs,
              # 3rd key (only for norm) in L_NORM_VALS

# ---------------------- helper functions -------------------
def get_n_rand_bit_seqs(seq_length, n, seed=0):
    """
    Returns a list of n pseudorandom but reproducible
    bit sequences of length seq_length (in bits)

    :param seq_length : length of the random sequence to generate (bits)
    :param n : number of sequences to produce
    :param seed : the seed for our pseudorandom number generation (can be provided)
                  for reproducibility

    """
    pass

for bit_num in SEQUENCE_LENGTHS:
    # get NUM_SEQS_OF_FIXED_LENGTH bit sequences of that length
    bit_seqs = get_n_rand_bit_seqs(bit_num, NUM_SEQS_OF_FIXED_LENGTH)

    for weighting_coef in WEIGHTING_COEFS:
        for l_norm in L_NORM_VALS:
            pass # run l-norm based tests
        pass # run BER
