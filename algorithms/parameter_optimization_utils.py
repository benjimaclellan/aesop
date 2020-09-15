import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
import copy
import random
import warnings
import time

from lib.analysis.hessian import function_wrapper
from .assets.functions import logbook_update, logbook_initialize

from problems.example.assets.additive_noise import AdditiveNoise

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
TODO: remove code duplicates between parameters_genetic_algorithm.py once functionality has been
      replicated here
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


def get_initial_population(graph, propagator, evaluator, n_pop, mutation_operator_dist, resample_per_individual=False):
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
        if (resample_per_individual):
            graph.resample_all_noise()

        parameters = graph.sample_parameters_to_list(probability_dist=mutation_operator_dist)
        
        # may or may not be necessary, depending on implementation changes
        graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)

        # graph.propagate(propagator)
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
    try:
        graph.distribute_parameters_from_list(individual_params, node_edge_index, parameter_index)

        return evaluator.evaluate_graph(graph, propagator)
    except RuntimeWarning as w:
        lower, upper = graph.get_parameter_bounds()
        print(f'lower bound: {lower}')
        print(f'upper bound: {upper}')
        print(f'individual params: {individual_params}')
        raise w


# -------------------- GA implementation ----------------------

def tuning_genetic_algorithm(graph, propagator, evaluator, n_generations=25,
                             n_population=64, rate_mut=0.9, rate_crx=0.9, noise=True,
                             crossover=crossover_singlepoint, mutation_operator_dist='uniform',
                             resample_per_individual=False, resample_period_gen=1, resample_period_batch=1,
                             optimize_top_X=0, optimization_batch_size=10, optimization_batch_num=25, verbose=False):
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

    :param resample_per_individual: If true, noise models of the graph are resampled for each individual 
                                    Then, resample_period_gen and resample_perod_Adam are ignored (since resampling occurs at most frequent possible interval)
                                    If False, resample_period_gen or resample_period_Adam govern resampling rate
    :param resample_period_gen: Period of resampling, in generations. If negative, never resample. Ignored if resample_per_individual is True
    :param resample_period_batch: Period of resampling per gradient optimization batch

    :param optimize_top_X: if 0, acts as a regular GA. Else, executes Adam gradient descent on the top x elements each generation
    :param optimization_batch_size: batch size for each round of optimization (if optimize_top_X > 0)
    :param optimization_batch_num : number of batches for optimization (if optimize_top_X > 0)

    :return: final_population (sorted), log_book
    :raise ValueError: if resampling period of optimization is not a multiple of batch size, and resample_per_individual is False
    """
    AdditiveNoise.simulate_with_noise = noise

    log, log_metrics = logbook_initialize()
    population, node_edge_index, parameter_index = \
        get_initial_population(graph, propagator, evaluator, n_population,
                               mutation_operator_dist, resample_per_individual=resample_per_individual)
    if verbose:
        print('optimize pop')
    if (optimize_top_X > 0):
        adam_logs = []

    # updates log book with initial population statistics
    logbook_update(0, population, log, log_metrics, verbose=verbose)

    #TODO: include and update logbook
    for generation_num in range(1, n_generations + 1):
        if verbose:
            print(f'Generation: {generation_num}')
        start_time = time.time() # saving runtime

        if (not resample_per_individual and resample_period_gen > 0 and generation_num % resample_period_gen == 0): # resample every resample_period_gen generations
            graph.resample_all_noise()

        # Cross-over
        for _ in range(np.floor(rate_crx * n_population).astype('int')):
            parent1, parent2 = [parent for (score, parent) in tuple(random.sample(population, 2))]
            children = crossover(parent1, parent2)
            for child in children:
    
                if (resample_per_individual):
                    graph.resample_all_noise()

                score = get_individual_score(graph, propagator, evaluator, child, node_edge_index, parameter_index)
                population.append((score, child))
        # mutation
        for _ in range(np.floor(rate_mut * n_population).astype('int')):
            parent = [parent for (score, parent) in tuple(random.sample(population, 1))][0]
            mut = graph.sample_parameters_to_list()
            child = mutation(parent, mut)

            if (resample_per_individual):
                graph.resample_all_noise()
    
            score = get_individual_score(graph, propagator, evaluator, child, node_edge_index, parameter_index)
            population.append((score, child))
               
        # sort population, and then take the best n_population individuals
        population.sort(reverse=False)
        if (optimize_top_X > 0):
            tuned_pop, adam_log = tuning_adam_gradient_descent(graph, propagator, evaluator, noise=noise,
                                                        n_pop=optimize_top_X, pop=population[0:optimize_top_X],
                                                        resample_per_individual=resample_per_individual,
                                                        resample_period=resample_period_batch,
                                                        n_batches=optimization_batch_num,
                                                        batch_size=optimization_batch_size)
            adam_logs.append(adam_log)
            
            population.extend(tuned_pop)
            population.sort(reverse=False)
        population = population[:-(len(population) - n_population) or None]
        
        # updates log book
        runtime = time.time() - start_time
        logbook_update(generation_num, population, log, log_metrics, runtime=runtime, verbose=verbose)
    
    if (optimize_top_X > 0):
        return population, log, adam_logs

    return population, log

# -------------------- Adam implementation ----------------------

def adam_function_wrapper(param_function):
    def _function(_params, _iter):
        return param_function(_params)
    
    return _function

def adam_bounded(lower_bounds, upper_bounds, grad, x, convergence_check_period=None,
                 convergence_thresh_abs=0.00085, callback=None, num_iters=100, step_size=0.001,
                 b1=0.9, b2=0.999, eps=10**-8,
                 m=None, v=None, verbose=True):
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

    # for testing purposes, we're going to allow these to be set, such that we can exit the function and restart where we ended
    if m is None:
        m = np.zeros(len(x))
    if v is None:
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
                if (verbose):
                    print(f'i: {i}, delta: {delta}, convergence_thresh_abs: {convergence_thresh_abs}')
                x = np.clip(x, a_min=lower_bounds, a_max=upper_bounds)
                return x, i, m, v

        x = np.clip(x, a_min=lower_bounds, a_max=upper_bounds)

    return x, num_iters, m, v


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

    params, iters_run, m, v = adam_bounded(lower_bounds, upper_bounds, fitness_grad, params,
                              convergence_check_period=convergence_check_period,
                              num_iters=adam_num_iters)

    return params, iters_run, m, v


def tuning_adam_gradient_descent(graph, propagator, evaluator, n_batches=125, batch_size=10, convergence_check_period=1,
                                 n_pop=64, pop=None, noise=True,
                                 resample_per_individual=False, resample_period=1,
                                 exclude_locked=True, verbose=True):
    """
    Performs adam gradient descent on `n_population` individuals. Data is logged and returned at the end

    :param graph: graph to tune
    :param propagator: propagator to use for graph evaluation
    :param evaluator: evaluator to use for graph evaluation
    :param n_batches: number of separate batches to run (log data gathered after each batch)
    :param batch_size: number of Adam iterations per batch
    :param convergence_check_period: the period at which to check for premature convergence
    :param n_pop: size of the population to tune
    :param pop: initial population to tune. If None, initial pop is generated by the function
    :param resample_per_individual: If true, noise models of the graph are resampled for each individual 
                                    Then, resample_period_gen and resample_perod_Adam are ignored (since resampling occurs at most frequent possible interval)
                                    If False, resample_period_gen or resample_period_Adam govern resampling rate
    :param resample_period: Period of resampling (in batches)
    :param exclude_locked: if True, exclude locked/constant parameters from derivative calculations. Else, include all parameters
    :param verbose: if True, print each log lines as they are generated. If else, don't.

    :raise ValueError: if len(pop) (assuming pop is not None) != n_pop

    :return optimized population, log
    """
    AdditiveNoise.simulate_with_noise = noise

    # setup initial population
    if (pop is None):
        pop, node_edge_index, parameter_index = get_initial_population(graph, propagator, evaluator, n_pop, 'uniform')
    else:
        if (len(pop) != n_pop):
            raise ValueError(f'Initial population provided does not have n_population (population size) {n_pop}')
        _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
    
    pop = [(score, np.array(params)) for (score, params) in pop]
    
    # setup the gradient function and bounds
    fitness_funct = function_wrapper(graph, propagator, evaluator, exclude_locked=exclude_locked)
    adam_fitness_funct = adam_function_wrapper(fitness_funct)
    fitness_grad = grad(adam_fitness_funct)

    lower_bounds, upper_bounds = graph.get_parameter_bounds()
    if (verbose):
        print(f'lower bounds: {lower_bounds}')
        print(f'upper bounds: {upper_bounds}')

    # setup logging, and update with initial population statistics
    log, log_metrics = logbook_initialize()
    logbook_update(0, pop, log, log_metrics, verbose=verbose)

    # setup tracking of individuals that have reached convergence
    has_converged = [False] * n_pop
    saved_m_v = [(None, None)] * n_pop

    # run each batch
    for batch in range(n_batches):
        if (verbose):
            print(f'Batch {batch}')

        if (not resample_per_individual and resample_period > 0 and batch % resample_period == 0):
            graph.resample_all_noise()

        start_time = time.time()
        for i in range(n_pop):
            if (resample_per_individual):
                graph.resample_all_noise()

            if (has_converged[i]):
                continue # if we've converged once, we skip future checks

            if (verbose):
                print(f'population #: {i}')
        
            tmp_param, actual_iters, m, v = adam_bounded(lower_bounds, upper_bounds, fitness_grad, pop[i][1],
                                                         convergence_check_period=convergence_check_period,
                                                         num_iters=batch_size, m=saved_m_v[i][0], v=saved_m_v[i][1], 
                                                         verbose=verbose)
            pop[i] = (None, tmp_param)
            saved_m_v[i] = (m, v)
            if (actual_iters != batch_size): # i.e. if it cut out early bc we've levelled out enough
                has_converged[i] = True

        runtime = time.time() - start_time
        
        # note that we don't count the population evaluation in the runtime, because it's not necessary for the optimization algorithm (unlike in GAs)
        # it's useful for logging, however 
    
        for i in range(n_pop):
            if (resample_per_individual): # doesn't help converges, but makes the quality criteria "fair" across the individuals
                graph.resample_all_noise()

            pop[i] = (get_individual_score(graph, propagator, evaluator, pop[i][1], node_edge_index, parameter_index), pop[i][1])
        
        # update log
        logbook_update(batch + 1, pop, log, log_metrics, runtime=runtime, verbose=verbose)
    
    # once batches are complete, cleanup
    pop.sort(reverse=False, key=lambda x: x[0]) # key is necessary in case score is the same for 2 elements
    pop = [(score, params.tolist()) for (score, params) in pop] # needed for compatibility with the GA operators
    return pop, log




