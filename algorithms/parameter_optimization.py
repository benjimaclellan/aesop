
import scipy.optimize
import cma
import autograd.numpy as np
import random
from algorithms.functions import logbook_update, logbook_initialize
import copy
import time
import pandas as pd
import itertools

"""
"""
LOG = []
ITER = itertools.count(0,1)

#  default use a minimizing function
def parameters_optimize(graph, x0=None, method='L-BFGS', verbose=False, log_callback=False, **kwargs):
    _, node_edge_index, parameter_index, lower_bounds, upper_bounds = graph.extract_parameters_to_list()


    def callback(xk):
        # scipy.optimize.minimize unfortunately does not pass score to callback, meaning to log score we need to re-calculate (doubles computation time)
        LOG.append({'iteration':next(ITER), 'score':graph.func(xk)})
        return

    def null_callback(xk):
        LOG.append({'iteration':next(ITER), 'score':-1})
        return

    if method == 'L-BFGS':
        if verbose: print("Parameter optimization: L-BFGS algorithm")
        t1 = time.process_time()
        res = scipy.optimize.minimize(graph.func, x0, method='L-BFGS-B',
                                      bounds=list(zip(lower_bounds, upper_bounds)),
                                      options={'disp': verbose, 'maxiter': 50},
                                      callback=(callback if log_callback else null_callback),
                                      jac=graph.grad)
        t2 = time.process_time()
        graph.distribute_parameters_from_list(res.x, node_edge_index, parameter_index)
        x = res.x

        log = pd.DataFrame(LOG)
        log['time'] = log['iteration']/np.max(log['iteration']) * (t2-t1)
        return graph, x, graph.func(x), log

    if method == 'L-BFGS+GA':
        if verbose: print("Parameter optimization: L-BFGS + GA algorithm")

        t1 = time.process_time()
        x, hof, log = parameters_genetic_algorithm(graph, n_generations=15, n_population=15, rate_mut=0.8,
                                                   rate_crx=0.8, verbose=verbose)
        t2 = time.process_time()

        total_log = pd.DataFrame(columns=['iteration','time','score'])
        total_log['iteration'] = log['generation']
        total_log['score'] = log['minimum']
        total_log['time'] = log['generation']/np.max(log['generation']) * (t2 - t1)
        total_log['method'] = 'GA'



        t1 = time.process_time()
        res = scipy.optimize.minimize(graph.func, x, method='L-BFGS-B',
                                      bounds=list(zip(lower_bounds, upper_bounds)),
                                      options={'disp': verbose, 'maxiter': 50},
                                      callback=callback,
                                      jac=graph.grad)
        t2 = time.process_time()
        log = pd.DataFrame(LOG)
        log['time'] = log['iteration'] / np.max(log['iteration']) * (t2 - t1) + total_log['time'].iloc[-1]
        log['method'] = 'L-BFGS'

        # print(total_log['time'].iloc[-1])
        total_log = pd.concat([total_log, log], sort=True)

        graph.distribute_parameters_from_list(res.x, node_edge_index, parameter_index)
        x = res.x
        return graph, x, graph.func(x), total_log

    elif method == 'CMA':
        # raise NotImplementedError("CMA is not implemented yet")
        if verbose: print("Parameter optimization: CMA algorithm")

        _, node_edge_index, parameter_index, lower_bounds, upper_bounds = graph.extract_parameters_to_list()
        es = cma.CMAEvolutionStrategy(x0, 0.999,
                                      {'verb_disp': int(verbose), 'maxfevals': 1000, 'bounds': [lower_bounds, upper_bounds]})
        es.optimize(graph.func)
        res = es.result
        x = res.xbest
        graph.distribute_parameters_from_list(x, node_edge_index, parameter_index)
        return graph, x, graph.func(x), res
        
    elif method == 'ADAM':
        if verbose: print("Parameter optimization: ADAM algorithm")
        x, num_iters, m, v = adam_bounded(np.array(lower_bounds), np.array(upper_bounds), graph.grad, np.array(x0),
                                          convergence_thresh_abs=0.00085, callback=None,
                                          num_iters=100, step_size=0.001, b1=0.9, b2=0.999,
                                          eps=10 ** -8, m=None, v=None, verbose=verbose)
        res = {"iterations":num_iters}
        graph.distribute_parameters_from_list(x, node_edge_index, parameter_index)
        return graph, x, graph.func(x), res

    if method == 'ADAM+GA':
        if verbose: print("Parameter optimization: GA + ADAM algorithm")
        x, hof, log = parameters_genetic_algorithm(graph, n_generations=15, n_population=15, rate_mut=0.9,
                                                   rate_crx=0.9, verbose=verbose)

        x, num_iters, m, v = adam_bounded(np.array(lower_bounds), np.array(upper_bounds), graph.grad, np.array(x0),
                                          convergence_thresh_abs=0.00085, callback=None,
                                          num_iters=100, step_size=0.001, b1=0.9, b2=0.999,
                                          eps=10 ** -8, m=None, v=None, verbose=verbose)
        res = {"iterations": num_iters}
        graph.distribute_parameters_from_list(x, node_edge_index, parameter_index)
        return graph, x, graph.func(x), res

    elif method == 'GA':
        if verbose: print("Parameter optimization: GA algorithm")

        x, hof, log = parameters_genetic_algorithm(graph, n_generations=15, n_population=15, rate_mut=0.9,
                                                   rate_crx=0.9,verbose=verbose)
        graph.distribute_parameters_from_list(x, node_edge_index, parameter_index)
        return graph, x, graph.func(x), log


    else:
        raise ModuleNotFoundError('This is not a defined minimization method')



def adam_bounded(lower_bounds, upper_bounds, grad, x, convergence_check_period=None,
                 convergence_thresh_abs=0.00085, callback=None, num_iters=100, step_size=0.001,
                 b1=0.9, b2=0.999, eps=10 ** -8,
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
    delta_arr = (upper_bounds - lower_bounds) * 10 ** -8  # some parameters are NOT differentiable on the boundary so we just avoid that...

    lower_bounds = lower_bounds + delta_arr
    upper_bounds = upper_bounds - delta_arr

    if (convergence_check_period is None):
        convergence_check_period = num_iters + 1  # i.e. will never check for convergence

    # for testing purposes, we're going to allow these to be set, such that we can exit the function and restart where we ended
    if m is None:
        m = np.zeros(len(x))
    if v is None:
        v = np.zeros(len(x))

    for i in range(num_iters):
        g = grad(x)
        if callback: callback(x, i, g)
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)

        if (i % convergence_check_period == convergence_check_period - 1):
            delta = np.linalg.norm(step_size * mhat / (np.sqrt(vhat) + eps))
            if (delta < convergence_thresh_abs):  # return early if the function meets our threshold of convergence
                if (verbose):
                    print(f'i: {i}, delta: {delta}, convergence_thresh_abs: {convergence_thresh_abs}')
                x = np.clip(x, a_min=lower_bounds, a_max=upper_bounds)
                return x, i, m, v

        x = np.clip(x, a_min=lower_bounds, a_max=upper_bounds)

    return x, num_iters, m, v



#
def parameters_genetic_algorithm(graph, n_generations = 25, n_population = 25, rate_mut = 0.9, rate_crx = 0.9, verbose=False):
    # hyper-parameters, will later be added as function arguments to change dynamically
    crossover = crossover_singlepoint
    mutation_operator = 'uniform'
    mut_kwargs = {}
    log, log_metrics = logbook_initialize()
    n_cores = 6
    new_individuals_divisor = 20

    # first we grab the information we will need about the parameters before running
    _, node_edge_index, parameter_index, lower_bounds, upper_bounds = graph.extract_parameters_to_list()

    # create initial population
    population = []
    for individual in range(n_population):
        x = graph.sample_parameters_to_list(probability_dist=mutation_operator, **mut_kwargs)
        score = graph.func(x)
        population.append((score, x))

    # updates log book with initial population statistics
    logbook_update(0, population, log, log_metrics, verbose=verbose)
    # update HoF status from the initial population, so we can compare in subsequent generations
    hof = sorted(population, reverse=False)[0]

    # loop through generations, applying evolution operators
    for generation in range(1, n_generations + 1):
        # *** here we evolve the population ***
        # we cross-over and mutate
        for child_i in range(np.floor(rate_crx * n_population).astype('int')):
            parent1, parent2 = [parent for (score, parent) in tuple(random.sample(population, 2))]
            children = crossover(parent1, parent2)
            for child in children:
                #     if np.random.rand() < rate_mut:
                #         mutant = graph.sample_parameters_to_list()  # this is a completely random individual which we can use
                #         child = mutation(child, mutant)
                population.append((None, child))
        for child_i in range(np.floor(rate_mut * n_population).astype('int')):
            parent = [parent for (score, parent) in tuple(random.sample(population, 1))][0]
            mutant = graph.sample_parameters_to_list()  # this is a completely random individual which we can use
            child = mutation(parent, mutant)
            population.append((None, child))

        for i in range(n_population // new_individuals_divisor):  # TODO: add a decay factor?
            # generate a few random new ppl to keep our populations spry and non-convergent
            x = graph.sample_parameters_to_list(probability_dist=mutation_operator, **mut_kwargs)
            score = graph.func(x)
            population.append((score, x))

        # loop through population and update scores for evolved individuals
        for i, (score, individual) in enumerate(population):
            if score is None:
                # then we score the ones which haven't been scored
                score = graph.func(individual)
                population[i] = (score, individual)

        # sort the population, and remove the worst performing individuals to keep population size consistent
        population.sort(reverse=False)
        population = population[
                     :-(len(population) - n_population) or None]  # remove last N worst performing individuals

        # update HoF status
        if population[0][0] < hof[0]:
            hof = population[0]  # set hof to best of this generation if it is better thant he existing hof

        # updates log book
        logbook_update(generation, population, log, log_metrics, verbose=verbose)

    return hof[1], hof[0], log


def crossover_singlepoint(parent1, parent2, **kwargs):
    try:
        crx_point = np.random.randint(1, len(parent1))
        child1 = copy.deepcopy(parent1[:crx_point] + parent2[crx_point:])
        child2 = copy.deepcopy(parent2[:crx_point] + parent1[crx_point:])
        return child1, child2
    except:
        return parent1, parent2


def crossover_doublepoint(parent1, parent2, **kwargs):
    crx_points = tuple(np.random.randint(2, len(parent1), 2))
    crx_point1, crx_point2 = min(crx_points), max(crx_points)
    child1 = copy.deepcopy(parent1[:crx_point1] + parent2[crx_point1:crx_point2] + parent1[crx_point2:])
    child2 = copy.deepcopy(parent2[:crx_point1] + parent1[crx_point1:crx_point2] + parent2[crx_point2:])
    return child1, child2


def mutation(parent, mutant):
    """ mutation evolution operator """

    num = random.randint(1, len(mutant))
    inds = set(np.random.choice(list(range(0, len(mutant))), num, replace=False))

    # mut_point = np.random.randint(1, len(parent))
    # parent[mut_point] = copy.deepcopy(mutant[mut_point])

    child = [mut if i in inds else par for i, (par, mut) in enumerate(zip(parent, mutant))]

    return child