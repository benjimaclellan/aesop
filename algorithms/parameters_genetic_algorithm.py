
"""


"""

import random
import copy

from config.config import np
from .assets.functions import logbook_update, logbook_initialize

def parameters_genetic_algorithm(graph, propagator, evaluator):
    # hyper-parameters, will later be added as function arguments to change dynamically
    n_generations = 75
    n_population = 100
    rate_mut = 0.5
    rate_crx = 0.3
    crossover = crossover_singlepoint
    mutation_operator = 'uniform'
    mut_kwargs = {}
    log, log_metrics = logbook_initialize()
    verbose = True

    # first we grab the information we will need about the parameters before running
    _, node_edge_index, parameter_index, lower_bounds, upper_bounds = graph.extract_parameters_to_list()

    # create initial population
    population = []
    for individual in range(n_population):
        parameters, *_ = graph.sample_parameters_to_list(probability_dist=mutation_operator, **mut_kwargs)
        graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)  # this probably isn't necessary, but if API changes it  may be
        graph.propagate(propagator)
        score = evaluator.evaluate_graph(graph, propagator)
        population.append((score, parameters))

    # updates log book with initial population statistics
    logbook_update(0, population, log, log_metrics, verbose=verbose)
    # update HoF status from the initial population, so we can compare in subsequent generations
    hof = sorted(population, reverse=False)[0]

    # loop through generations, applying evolution operators
    for generation in range(1, n_generations+1):

        # *** here we evolve the population ***
        # we cross-over and mutate
        for child_i in range(np.floor(rate_crx * n_population).astype('int')):
            parent1, parent2 = [parent for (score, parent) in tuple(random.sample(population, 2))]
            children = crossover(parent1, parent2)
            for child in children:
                if np.random.rand() < rate_mut:
                    mutation(child)
                population.append((None, child))

        # loop through population and update scores for evolved individuals
        for i, (score, individual) in enumerate(population):
            if score is None:
                # then we score the ones which haven't been scored
                graph.distribute_parameters_from_list(individual, node_edge_index, parameter_index)
                graph.propagate(propagator)
                score = evaluator.evaluate_graph(graph, propagator)
                population[i] = (score, individual)

        # sort the population, and remove the worst performing individuals to keep population size consistent
        population.sort(reverse=False)
        population = population[:-(len(population)-n_population) or None]  # remove last N worst performing individuals

        # update HoF status
        if population[0][0] > hof[0]:
            hof = population[0]  # set hof to best of this generation if it is better thant he existing hof

        # updates log book
        logbook_update(generation, population, log, log_metrics, verbose=verbose)

    return hof[1], hof[0], log


def crossover_singlepoint(parent1, parent2, **kwargs):
    crx_point = np.random.randint(1, len(parent1))
    child1 = copy.deepcopy(parent1[:crx_point] + parent2[crx_point:])
    child2 = copy.deepcopy(parent2[:crx_point] + parent1[crx_point:])
    return child1, child2

def crossover_doublepoint(parent1, parent2, **kwargs):
    crx_points = tuple(np.random.randint(2, len(parent1), 2))
    crx_point1, crx_point2 = min(crx_points), max(crx_points)
    child1 = copy.deepcopy(parent1[:crx_point1] + parent2[crx_point1:crx_point2] + parent1[crx_point2:])
    child2 = copy.deepcopy(parent2[:crx_point1] + parent1[crx_point1:crx_point2] + parent2[crx_point2:])
    return child1, child2


def mutation(parent):
    # TODO: add this
    mut_point = np.random.randint(1, len(parent))
    parent[mut_point] += np.random.normal(parent[mut_point], 10)
    return parent