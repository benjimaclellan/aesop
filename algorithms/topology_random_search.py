"""
Random search for topology for benchmarking
"""

import numpy as np
import datetime

from .assets.functions import logbook_update, logbook_initialize

from .parameters_genetic_algorithm import parameters_genetic_algorithm

def topology_random_search(graph, propagator, evaluator, evolver):
    n_generations = 5
    n_population = 5
    verbose = True
    hof = (None, None)
    log, log_metrics = logbook_initialize()
    _, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()

    for generation in range(n_generations+1):
        population = []
        for individual in range(n_population):
            print('generation {} of {} | individual {} of {} | started at {}'.format(generation, n_generations, individual, n_population, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            graph = evolver.random_graph(graph, evaluator)
            _, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()

            parameters, score, log = parameters_genetic_algorithm(graph, propagator, evaluator)

            graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)
            population.append((score, graph))

        population.sort(reverse = False)  # we sort ascending, and take first (this is the minimum, as we minimizing)
        logbook_update(generation, population, log, log_metrics, verbose=verbose)

        # update HoF status
        if hof[0] is None:
            hof = population[0]  # set hof for the first time
        elif population[0][0] > hof[0]:
            hof = population[0]  # set hof to best of this generation if it is better thant he existing hof

    return hof[1], hof[0], log

