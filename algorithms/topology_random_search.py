"""
Random search for topology for benchmarking
"""

import numpy as np
from .assets.functions import logbook_update, logbook_initialize

def parameters_random_search(graph, propagator, evaluator):
    n_generations = 75
    n_population = 100
    log, log_metrics = logbook_initialize()
    verbose = True
    hof = (None, None)
    _, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()

    for generation in range(n_generations+1):
        population = []
        for individual in range(n_population):
            parameters = graph.sample_parameters_to_list(probability_dist='uniform', **{'triangle_width': 0.05})

            graph.distribute_parameters_from_list(parameters, node_edge_index,
                                                  parameter_index)
            graph.propagate(propagator)
            score = evaluator.evaluate_graph(graph, propagator)
            population.append((score, parameters))

        population.sort(reverse = False)  # we sort ascending, and take first (this is the minimum, as we minimizing)
        logbook_update(generation, population, log, log_metrics, verbose=verbose)

        # update HoF status
        if hof[0] is None:
            hof = population[0]  # set hof for the first time
        elif population[0][0] > hof[0]:
            hof = population[0]  # set hof to best of this generation if it is better thant he existing hof

    return hof[1], hof[0], log

