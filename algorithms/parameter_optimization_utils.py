import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
import copy
import random
import warnings

from lib.analysis.hessian import function_wrapper
from .assets.functions import logbook_update, logbook_initialize

"""
Util objective:

To optimize performance following a given fitness function, using gradient descent algorithms (Adam, particularly)
and genetic algorithms. This util is only intended for tunable parameter optimization (i.e does not include changes 
in the computational graph structure).

Note that we sbashbashall want to compare effects of changes along 2 axes:
1. Batch size for gradient descent
2. Initialisation of parameters

Adam is proven to reach the global minimum only for convex functions: otherwise it is guaranteed to reach a local minimum.
Since our fitness function is not provably convex, we will offer configurable options to find starting points for Adam. 
Options can include:
1. Random initialisation of N individuals (each of which may be optimize)
2. Running a genetic algorithm for G generations, with a population of N individuals
3. Running a generatic algorithm for G generations, with a population of N invididuals,
   and use the final population as the set of starting points for the Adam optimization

Author: Julie Belleville
Resources: Genetic algorithm adapted from `parameters_genetic_algorithm.py`
"""

"""
TODO: allow multiprocessing (multithread it!)
TODO: consider whether we'd like to test other autograd-provided optimizers?
TODO: test boundary setting methods (rigging the fitness function to have a steep slope at
      boundaries, or manually removing components of the vector which would violate boundaries)
TODO: remove code duplicates between parameters_genetic_algorithm.py once functionality has been
      replicated here
TODO: decide on termination condition for Adam (score is approx static? Number of iterations?)
TODO: investigate those invalid values in sqrt / power runtime warnings
"""

# -------------------- Helper functions for GA ----------------------

def crossover_singlepoint(parent1, parent2, **kwargs):
    """ Crossover operator for GA """
    try:
        crx_point = np.random.randint(1, len(parent1))
        child1 = copy.deepcopy(parent1[:crx_point] + parent2[crx_point:])
        child2 = copy.deepcopy(parent2[:crx_point] + parent1[crx_point:])
        return child1, child2
    except:
        return parent1, parent2


def crossover_doublepoint(parent1, parent2, **kwargs):
    """ Other crossover operator """
    crx_points = tuple(np.random.randint(2, len(parent1), 2))
    crx_point1, crx_point2 = min(crx_points), max(crx_points)
    child1 = copy.deepcopy(parent1[:crx_point1] + parent2[crx_point1:crx_point2] + parent1[crx_point2:])
    child2 = copy.deepcopy(parent2[:crx_point1] + parent1[crx_point1:crx_point2] + parent2[crx_point2:])
    return child1, child2


def mutation(parent, mutant):
    """ mutation evolution operator """
    num = np.random.randint(1, len(mutant))
    inds = set(np.random.choice(list(range(0,len(mutant))), num, replace=False))


    # mut_point = np.random.randint(1, len(parent))
    # parent[mut_point] = copy.deepcopy(mutant[mut_point])

    child = [mut if i in inds else par for i, (par, mut) in enumerate(zip(parent, mutant))]

    return child


def get_initial_population(graph, propagator, evaluator, n_pop, mutation_operator_dist):
    """
    Initializes a random population of the same topology,
    but different tuning parameter values

    :param graph : topology which all individuals of the population have
    :param propagator : propagator with which initial scores are determined
    :param evaluator : evaluator with which initial scores are determined
    :param n_pop : size of the population being initialised
    :return : randomly initialised population, node edge indices, parameter indices
    """
    # first we grab the information we will need about the parameters before running
    _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
    
    population = []
    for _ in range(n_pop):
        parameters = graph.sample_parameters_to_list(probability_dist=mutation_operator_dist)
        
        # may or may not be necessary, depending on implementation changes
        graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)

        graph.propagate(propagator)
        score = evaluator.evaluate_graph(graph, propagator)
        population.append((score, parameters))
    
    return population, node_edge_index, parameter_index

def get_individual_score(graph, propagator, evaluator, individual_params, node_edge_index, parameter_index):
    """
    Gets the score of an individual

    :param graph : topology of the individual
    :param propagator : propagator with which the score is determined
    :param evaluator : evaluator with which the score is determined
    :param individual : the set of parameters describing the individual with graph's topology
    :param node_edge_index : indices of the node edge (matches with the individual params)
    :param param_Edge_index : ditto

    :return: the score received by the individual with this evaluator
    """
    graph.distribute_parameters_from_list(individual_params, node_edge_index, parameter_index)
    graph.propagate(propagator)
    return evaluator.evaluate_graph(graph, propagator)


# -------------------- GA implementation ----------------------

def tuning_genetic_algorithm(graph, propagator, evaluator, n_generations=25,
                             n_population=64, rate_mut=0.9, rate_crx=0.9, 
                             crossover=crossover_singlepoint, mutation_operator_dist='uniform',
                             verbose=False):
    """
    Genetic algorithm for tuning parameters in a set topology. Parametrised version
    of the function implemented by @benjimaclellan

    :param graph: graph to tune
    :param propagator: propagator to use for graph evaluation
    :param evaluator: evaluator to use for graph evaluation
    :param n_generations: number of generations to the tuning genetic algorithm
    :param rate_mut: the rate of mutation
    :param rate_crx: the rate of crossover
    :param crossover: crossover function
    :param mutation_operator_dist: distribution of the mutation operator
    :return: final_population (sorted), log_book
    """
    log, log_metrics = logbook_initialize()
    population, node_edge_index, parameter_index = \
        get_initial_population(graph, propagator, evaluator, n_population,
                               mutation_operator_dist)
    
    # updates log book with initial population statistics
    logbook_update(0, population, log, log_metrics, verbose=verbose)

    #TODO: include and update logbook
    for generation_num in range(1, n_generations + 1):
        # Cross-over
        for _ in range(np.floor(rate_crx * n_population).astype('int')):
            parent1, parent2 = [parent for (score, parent) in tuple(random.sample(population, 2))]
            children = crossover(parent1, parent2)
            for child in children:
                score = get_individual_score(graph, propagator, evaluator, child, node_edge_index, parameter_index)
                population.append((score, child))
        # mutation
        for _ in range(np.floor(rate_mut * n_population).astype('int')):
            parent = [parent for (score, parent) in tuple(random.sample(population, 1))][0]
            mut = graph.sample_parameters_to_list()
            child = mutation(parent, mut)
            score = get_individual_score(graph, propagator, evaluator, child, node_edge_index, parameter_index)
            population.append((score, child))
       
        # TODO: consider entirely new individuals to prevent early convergence?
        
        # sort population, and then take the best n_population individuals
        population.sort(reverse=False)
        population = population[:-(len(population) - n_population) or None]
        
        # updates log book
        logbook_update(generation_num, population, log, log_metrics, verbose=verbose)
    
    return population, log, log_metrics

# -------------------- Adam implementation ----------------------

def adam_function_wrapper(param_function):
    def _function(_params, _iter):
        return param_function(_params)
    
    return _function

def adam_bounded(lower_bounds, upper_bounds, grad, x, convergence_check_period=None,
                 convergence_thresh_abs=0.00085, callback=None, num_iters=100, step_size=0.001,
                 b1=0.9, b2=0.999, eps=10**-8):
    """
    Adam, as implemented by the autograd library: https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py
    
    Modified via the gradient projection method (in order to keep parameter values within physically required bounds) and convergence check. At each
    iteration, the gradient is applied and then the resulting vector is clipped to within its bounds. Every
    `convergence_check_period` iterations, we check whether the magnitude of the most recent change is smaller than a threshold. If so, we return

    TODO: Reread algorithm and determine how to handle the moment estimates because I feel like that... might be a thing

    Description from the autograd team: Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.
    
    :param lower_bounds : dimension must match x. lower_bounds[i] is the lower bound of parameter x[i]
    :param upper_bounds : dimension must match x. upper_bounds[i] is the upper bound of parameter x[i]
    :param x : x should be a np array!!

    :return <new parameters>, last iteration number (num_iters if no early termination)
    """
    delta_arr = (upper_bounds - lower_bounds) * 10**-8 # some parameters are NOT differentiable on the boundary so we just avoid that...

    lower_bounds = lower_bounds + delta_arr
    upper_bounds = upper_bounds - delta_arr

    if (convergence_check_period is None):
        convergence_check_period = num_iters + 1 # i.e. will never check for convergence

    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)

        if (i % convergence_check_period == convergence_check_period - 1):
            delta = np.linalg.norm(step_size*mhat/(np.sqrt(vhat) + eps))
            if (delta < convergence_thresh_abs): # return early if the function meets our threshold of convergence
                print(f'i: {i}, delta: {delta}, convergence_thresh_abs: {convergence_thresh_abs}')
                return x, i

        x = np.clip(x, a_min=lower_bounds, a_max=upper_bounds)

    return x, num_iters


def adam_gradient_projection(graph, propagator, evaluator, params,
                             convergence_check_period=None,
                             adam_num_iters=100, exclude_locked=True):
    """
    Performs Adam gradient descent of `graph` parameters on a single graph topology,
    starting at `start_param`. This function does not take parameter bounds into account
    (i.e. assumes that the fitness function has the bounds built in)

    :pre : the evaluator `evaluate_graph` function is differentiable by autograd
    :param graph: graph to tune
    :param propagator: propagator to use for graph evaluation
    :param evaluator: evaluator to use for graph evaluation
    :param params: start value for parameters
    :param total_iters: temporary parameter to determine the number of times to run each adam optimization call
    :param adam_num_iters: parameter batch size for each Adam call (100 is the autograd default)
    #TODO: replace total_iters by some stability metric
    """

    fitness_funct = function_wrapper(graph, propagator, evaluator, exclude_locked=exclude_locked)
    adam_fitness_funct = adam_function_wrapper(fitness_funct)
    fitness_grad = grad(adam_fitness_funct)

    lower_bounds, upper_bounds = graph.get_parameter_bounds()

    params = adam_bounded(lower_bounds, upper_bounds, fitness_grad, params,
                          convergence_check_period=convergence_check_period,
                          num_iters=adam_num_iters)

    return params
