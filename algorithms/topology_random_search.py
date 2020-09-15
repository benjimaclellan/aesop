"""
Random search for topology for benchmarking
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt
import copy
import multiprocess as mp

from .assets.functions import logbook_update, logbook_initialize

from .parameter_builtin import parameters_optimize



def topology_random_search(graph, propagator, evaluator, evolver, io, multiprocess=False):
    n_generations = 2
    n_population = mp.cpu_count()
    # n_hof = 4
    verbose = True
    hof = (None, None)
    log, log_metrics = logbook_initialize()
    save_all_graphs = True

    _, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    if multiprocess:
        pool = mp.Pool(mp.cpu_count())
        print('Starting multiprocessing with {} CPUs'.format(mp.cpu_count()))

    population = []
    for individual in range(n_population):
        population.append((None, copy.deepcopy(graph)))

    for generation in range(n_generations):
        print('Generation {}'.format(generation))
        # pass in a list, each element is a list, consisting of
        args = [(evolver, graph, evaluator, propagator) for (score, graph) in population]

        if multiprocess:
            population = pool.map(parameters_optimize_multiprocess, args)
        else:
            population = list(map(parameters_optimize_multiprocess, args))

        population.sort(reverse = False, key=lambda x: x[0])  # we sort ascending, and take first (this is the minimum, as we minimizing)

        # save all graphs if desired
        if save_all_graphs:
            for i, (_, graph_i) in enumerate(population):
                io.save_graph(graph=graph_i, filename=f"gen{generation}_{i}.pkl")

        # logbook_update(generation, population, log, log_metrics, verbose=verbose)

        # update HoF status
        if hof[0] is None:
            hof = population[0]  # set hof for the first time
        elif population[0][0] < hof[0]:
            hof = population[0]  # set hof to best of this generation if it is better thant he existing hof

    graph = hof[1]
    score = hof[0]
    return graph, score, log


def parameters_optimize_multiprocess(args):
    evolver = args[0]
    graph_individual = args[1]
    evaluator = args[2]
    propagator = args[3]
    while True:
        graph, evo_op_choice = evolver.evolve_graph(graph_individual, evaluator, propagator)
        x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
        if len(x0) == 0:
            continue
        else:
            break

    graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()

    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
    graph, parameters, score, log = parameters_optimize(graph, x0=x0, method='L-BFGS+GA', verbose=False)
    return score, graph