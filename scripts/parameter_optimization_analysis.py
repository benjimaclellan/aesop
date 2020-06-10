import autograd.numpy as np
import matplotlib.pyplot as plt

from algorithms.parameter_optimization_utils import tuning_genetic_algorithm as parameters_genetic_algorithm
from algorithms.parameter_optimization_utils import adam_ignore_bounds, get_initial_population, get_individual_score

from problems.example.evaluator_subclasses.weighted_evaluators.case_evaluator_norm import NormCaseEvaluator
from problems.example.evaluator_subclasses.weighted_evaluators.case_evaluator_BER import BERCaseEvaluator
from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import power_

from algorithms.assets.functions import logbook_update, logbook_initialize


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
TODO: figure out what would be good propagator values (window, # samples) to test bc tbh I have no idea
TODO: normalize so that smallest power is shifted to 0?
TODO: what if we reward lack of up and down within a given bit or stretch of "high"? (a sort of derivative grading scheme?)

Observations:
1. We're lacking incentive on the quick up-downs bc it's worth just saying screw it?
2. The populations are converging weirdly fast: can we give it some intensive to explore more?
"""

DISPLAY_GRAPHICS = True
# ---- Defining testing variables
SEQUENCE_LENGTHS = [8, 32, 128]
L_NORM_VALS = [1, 2]
WEIGHTING_EXP = [0, 2, 4, 8, 16]
NUM_SEQS_OF_FIXED_LENGTH = 3 # how many sequences of the same length to test
BIT_WIDTH = 2**5 # powers of 2 I guess

log_dict = {} # build a dictionary structure to hold
propagator = Propagator(window_t = 1e-8, n_samples = 2**14, central_wl=1.55e-6)

# ---------------------- helper functions -------------------

# ---- Sourcing needed objects

def get_n_rand_bit_seqs(seq_length, n):
    """
    Returns a list of n pseudorandom but reproducible
    bit sequences of length seq_length (in bits)

    :param seq_length : length of the random sequence to generate (bits)
    :param n : number of sequences to produce
    :param seed : the seed for our pseudorandom number generation (can be provided)
                  for reproducibility

    """
    bit_sequences = [None] * n
    for i in range(n):
        bit_sequences[i] = np.random.choice([0, 1], p=[0.5, 0.5], size=seq_length).astype(bool)
    return bit_sequences


def get_graph():
    """
    Returns the default graph for testing, with fixed topology at this time
    """
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1, 'central_wl': 1.55e-6}),
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: WaveShaper(),
             3: DelayLine(),
             4: MeasurementDevice()
             }
    edges = [(0,1, CorningFiber(parameters=[0])),
             (1,2, CorningFiber(parameters=[0])),
             (2,3),
             (3,4)]

    graph = Graph(nodes, edges, propagate_on_edges = True)
    graph.assert_number_of_edges()
    return graph

# ---- Visualisation
def display_output_bit_seq(bit_seq, target_output, graph, graph_parameters, propagator, title=""):
    if DISPLAY_GRAPHICS:
        x = propagator.t
        pnts_displayed = bit_seq.shape[0] * BIT_WIDTH * 5 # just so visualisation isn't too crowded

        print(f'graph parameters: {graph_parameters}')
        _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
        graph.distribute_parameters_from_list(graph_parameters, node_edge_index, parameter_index)
        graph.propagate(propagator)
        state = graph.nodes[len(graph.nodes) - 1]['states'][0]
        actual_output = power_(state)
        
        actual_normalized = (actual_output - np.min(actual_output)) / (np.max(actual_output) - np.min(actual_output))

        fig, ax = plt.subplots()    
        ax.plot(x[0:pnts_displayed], target_output[0:pnts_displayed], label='target', color='r', lw=1)
        ax.plot(x[0:pnts_displayed], actual_normalized[0:pnts_displayed], label='actual', color='b', lw=1)
        ax.xaxis.set_major_locator(plt.MultipleLocator(BIT_WIDTH * propagator.dt * bit_seq.shape[0]))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(BIT_WIDTH * propagator.dt))
        ax.grid(which='minor', alpha=0.5)

        ax.legend()
        plt.title(title)
        plt.show()


def display_output_sawtooth(evaluator, graph, graph_parameters, propagator, title="Sawtooth"):
    if DISPLAY_GRAPHICS:
        pnts_displayed = propagator.t.shape[0] // 10 # just so visualisation isn't too crowded

        _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
        graph.distribute_parameters_from_list(graph_parameters, node_edge_index, parameter_index)
        graph.propagate(propagator)
        state = graph.nodes[len(graph.nodes) - 1]['states'][0]
        actual_output = power_(state)
        
        # actual_normalized = (actual_output - np.min(actual_output)) / (np.max(actual_output) - np.min(actual_output))
        actual_normalized = actual_output / np.max(actual_output)
        expected_normalised = evaluator.target / np.max(evaluator.target)

        _, ax = plt.subplots()
        ax.plot(propagator.t[0:pnts_displayed], expected_normalised[0:pnts_displayed], label='target', color='r', lw=1)
        ax.plot(propagator.t[0:pnts_displayed], actual_normalized[0:pnts_displayed], label='actual', color='b', lw=1)
        ax.legend()
        plt.title(title)
        plt.show()


# ---- Running relevant tests

# -- Case fitness analysis, GA
def run_all_lnorm(bit_seqs):
    """
    Runs all l-norm tests, and returns the relevant logs and data

    :param bit_seqs: list of bit sequences against which to test convergences
    :return : a dictionary containing all l-norm data for a given bit sequence length
    """
    print("l-norm results")
    for l_norm in L_NORM_VALS:
        for exp in WEIGHTING_EXP:
            for bit_seq in bit_seqs:
                evaluator = NormCaseEvaluator(propagator, bit_seq, BIT_WIDTH,
                                              norm=l_norm, weighting_exponent=exp)
                graph = get_graph()
                population, log = parameters_genetic_algorithm(graph, propagator, evaluator)
                score, individual_params = population[0]
                print(f"norm: {l_norm}, weight exp: {exp}:")
                print(log)
                display_output_bit_seq(bit_seq, evaluator._target, graph, individual_params, propagator, title=f'top individual, score: {score}')


def run_all_BER(bit_seqs):
    #TODO: create a propagator
    print("BER Results")
    for exp in WEIGHTING_EXP:
        for bit_seq in bit_seqs:
            evaluator = BERCaseEvaluator(propagator, bit_seq, BIT_WIDTH, weighting_exponent=exp)
            graph = get_graph()
            population, log = parameters_genetic_algorithm(graph, propagator, evaluator)
            score, individual_params = population[0]
            print(f"weight exp: {exp}:")
            print(log)
            display_output_bit_seq(bit_seq, evaluator._target, graph, individual_params, propagator, title=f'top individual, score: {score}')

# -- GA vs Adam comparison
# TODO: compare runtimes, and result qualities
# TODO: save some logs

def run_GA_only(k_gen, n_pop, graph, propagator, evaluator):
    """
    For benchmarking: observe the results after k generation of the GA (where N provides convergence),
                      on a population of size n
    :param k_gen : total number of generations in the GA
    :param n_pop : size n of the population in use
    :param graph, propagator, evaluator : really the usual stuff
    """
    population, log, log_metrics = parameters_genetic_algorithm(graph, propagator, evaluator, n_population=n_pop, n_generations=k_gen)
    score, individual_params = population[0]
    print('GA only')
    print(log)
    display_output_sawtooth(evaluator, graph, individual_params, propagator, title=f'Sawtooth top individual GA, score: {score}')
    return population, log, log_metrics


def _helper_adam(n_pop, graph, propagator, evaluator, population, log, log_metrics, adam_funct):
    new_pop = []
    
    _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
    nan_params = 0

    for (_, params) in population:
        params = adam_funct(graph, propagator, evaluator, params)
        if (not np.isnan(params).any()):
            score = get_individual_score(graph, propagator, evaluator, params, node_edge_index, parameter_index)
            new_pop.append((score, params))
        else:
            nan_params += 1
    
    new_pop.sort(reverse=False)
    score, individual_params = new_pop[0]
    logbook_update(0, new_pop, log, log_metrics, verbose=False)
    
    print(f'Ended up with NaN: {100 * nan_params / len(new_pop)}%')
    display_output_sawtooth(evaluator, graph, individual_params, propagator, title=f'Sawtooth top individual Adam, score: {score}')
    return new_pop, log, log_metrics


def run_random_generation_adam(n_pop, graph, propagator, evaluator, adam_funct=adam_ignore_bounds):
    """
    Observe scores from Adam gradient descent, starting from a random pool of parameters 

    :param n_pop : population size (i.e. number of different start parameters)
    :param graph, propagator, evaluator : really the usual stuff
    :param adam_funct : the Adam function to use
    """
    pop, _, _ = get_initial_population(graph, propagator, evaluator, n_pop, 'uniform') # we want the scores for logging purposes
    log, log_metrics = logbook_initialize()

    return _helper_adam(n_pop, graph, propagator, evaluator, pop, log, log_metrics, adam_funct), log, log_metrics


def run_GA_Adam(k_gen, n_pop, graph, propagator, evaluator, n_adam_optimize=None, adam_funct=adam_ignore_bounds):
    """
    Observe scores from Adam gradient descent, starting from parameters selected by our GA
    
    :param k_gen : total number of generations in the GA
    :param n_pop : size n of the population in use
    :param graph, propagator, evaluator : really the usual stuff
    :param n_adam_optimize : the number of the individuals to optimize via Adam. By default, optimize all
    :param adam_funct : the Adam function to use
    """
    if n_adam_optimize is None:
        n_adam_optimize = n_pop

    population, log, log_metrics = run_GA_only(k_gen, n_pop, graph, propagator, evaluator)

    optimal_pop = population[0:n_adam_optimize]

    new_pop = _helper_adam(n_pop, graph, propagator, evaluator, optimal_pop, log, log_metrics, adam_funct)
    return new_pop, log, log_metrics

# --------------------------- Data generation scripts -----------------------

# ---- Case fitness Analysis GA
def run_case_fitness_analysis_benchmarking():
    # -- Data generation
    np.random.seed(257)
    for bit_num in SEQUENCE_LENGTHS:
        bit_seqs = get_n_rand_bit_seqs(bit_num, NUM_SEQS_OF_FIXED_LENGTH)
        run_all_lnorm(bit_seqs)
        run_all_BER(bit_seqs)

    # TODO: -- Saving Data (data analysis can go in a separate script methinks)

# ---- Benchmarking GA (Sawtooth), arbitrarily initialised Adam, GA initialised Adam
def optimization_comparison_GA_Adam_both():
    graph = get_graph()
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    # print("GA")
    # run_GA_only(25, 50, graph, propagator, evaluator)
    # print("Adam")
    # run_random_generation_adam(50, graph, propagator, evaluator)
    print("GA + Adam")
    run_GA_Adam(25, 50, graph, propagator, evaluator, n_adam_optimize=5)
