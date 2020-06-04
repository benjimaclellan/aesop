from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm
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
TODO: new graph needed each time
"""

# ---------------------- helper functions -------------------

# ---- Sourcing needed objects

def get_n_rand_bit_seqs(seq_length, n, seed=0):
    """
    Returns a list of n pseudorandom but reproducible
    bit sequences of length seq_length (in bits)

    :param seq_length : length of the random sequence to generate (bits)
    :param n : number of sequences to produce
    :param seed : the seed for our pseudorandom number generation (can be provided)
                  for reproducibility

    """
    # TODO: implement
    return None

def get_graph():
    """
    Returns the default graph for testing, with fixed topology at this time
    """
    # TODO: implement
    return None

# ---- Running relevant tests

def run_all_lnorm(bit_seqs):
    """
    Runs all l-norm tests, and returns the relevant logs and data

    :param bit_seqs: list of bit sequences against which to test convergences
    :return : a dictionary containing all l-norm data for a given bit sequence length
    """
    #TODO: create a propagator
    for l_norm in L_NORM_VALS:
        for exp in WEIGHTING_EXP:
            for bit_seq in bit_seqs:
                # TODO: create new evaluator
                # TODO: source a new graph
                # TODO: run parameters_G_A and save all relevant data
                pass


def run_all_BER(bit_seqs):
    #TODO: create a propagator
    for exp in WEIGHTING_EXP:
        for bit_seq in bit_seqs:
            # TODO: create new evaluator
            # TODO: source a new graph
            # TODO: run parameters_G_A and save all relevant data
            pass

# --------------------------- Data generation script -----------------------


# ---- Defining testing variables
SEQUENCE_LENGTHS = [8, 32, 128]
L_NORM_VALS = [1, 2]
WEIGHTING_EXP = [0, 2, 4, 8, 16]
NUM_SEQS_OF_FIXED_LENGTH = 3 # how many sequences of the same length to test
BIT_WIDTH = 8 # powers of 2 I guess

log_dict = {} # build a dictionary structure to hold

# ---- Data generation

for bit_num in SEQUENCE_LENGTHS:
    # get NUM_SEQS_OF_FIXED_LENGTH bit sequences of that length
    bit_seqs = get_n_rand_bit_seqs(bit_num, NUM_SEQS_OF_FIXED_LENGTH)
    run_all_lnorm(bit_seqs)
    run_all_BER(bit_seqs)

# ---- Saving Data (data analysis can go in a separate script methinks)

pass