import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
import copy

import problems.example.evaluator
import problems.example.propagator
import problems.example.graph
from lib.analysis.hessian import function_wrapper

"""
Util objective:

To optimize performance following a given fitness function, using gradient descent algorithms (Adam, particularly)
and genetic algorithms. This util is only intended for tunable parameter optimization (i.e does not include changes 
in the computational graph structure).

Note that we shall want to compare effects of changes along 2 axes:
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
TODO: add logs
TODO: allow multiprocessing (multithread it!)
TODO: consider whether we'd like to test other autograd-provided optimizers?
TODO: test boundary setting methods (rigging the fitness function to have a steep slope at
      boundaries, or manually removing components of the vector which would violate boundaries)
TODO: remove code duplicates between parameters_genetic_algorithm.py once functionality has been
      replicated here
TODO: ponder refactoring into object-oriented (reducing the passed parameters a bit?)
TODO: decide on termination condition for Adam (score is approx static? Number of iterations?)
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
        graph.distribute_parameters_from_lis(parameters, node_edge_index, parameter_index)

        graph.propagate(propagator)
        score = evaluator.evaluate_graph(graph, propagator)
        population.append((score, parameters))
    
    return population, node_edge_index, parameter_index


def get_individual_score(graph, propagator, evaluator, individual, node_edge_index, parameter_index):
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
    graph.distribute_parameters_from_list(individual, node_edge_index, parameter_index)
    graph.propagator(propagator)
    return evaluator.evaluate_graph(graph, propagator)


# -------------------- GA implementation ----------------------

def tuning_genetic_algorithm(graph, propagator, evaluator, n_generations=25,
                             n_population=64, rate_mut=0.9, rate_crx=0.9, 
                             crossover=crossover_singlepoint, mutation_operator_dist='uniform'):
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
    population, node_edge_index, parameter_index = \
        get_initial_population(graph, propagator, evaluator, n_population,
                               mutation_operator_dist)

    #TODO: include and update logbook
    for _ in range(n_generations):
        # Cross-over
        for i in range(np.floor(rate_crx * n_population).astype('int')):
            parent1, parent2 = [parent for (score, parent) in tuple(np.random.sample(population, 2))]
            children = crossover(parent1, parent2)
            for child in children:
                score = get_individual_score(graph, propagator,
                                            evaluator, child,
                                            node_edge_index,
                                            parameter_index)
                population.append((score, child))
        #mutation
        for i in range(np.floor(rate_mut * n_population).astype('int')):
            parent = [parent for (score, parent) in tuple(np.random.sample(population, 1))]
            mut = graph.sample_parameters_to_list()
            child = mutation(parent, mut)
            score = get_individual_score(graph, propagator,
                                         evaluator, child,
                                         node_edge_index, parameter_index)
            population.append((score, child))
        
        # TODO: consider entirely new individuals to prevent early convergence?
        # sort population, and then take the best n_population individuals
        population.sort(reverse=False)
        population = population[:-(len(population) - n_population) or None]
        #TODO: update logs
    
    return population #, log

# -------------------- Adam implementation ----------------------

def adam_ignore_bounds(graph, propagator, evaluator, params, total_iters=100, adam_num_iters=100):
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
    #TODO: replace num_iterations by some stability metric
    """
    fitness_funct = function_wrapper(graph, propagator, evaluator)
    fitness_grad = grad(fitness_funct)
    for _ in range(total_iters):
        params = adam(fitness_grad, params, num_iters=adam_num_iters)

    return params

    

"""
Plan: create another Adam function which wraps the 'evaluate_graph' fitness function such that the parameter
      bounds are enforced by a dramatic increase in function value at the parameter edges

      Also create an Adam function that just inspects each parameters and hard resets parameter A to its 
      max and min if it steps over in either direction
"""