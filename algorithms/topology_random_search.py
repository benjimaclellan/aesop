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

def init_hof(n_hof):
    hof = [(None, None) for i in range(n_hof)]
    return hof

def update_hof(hof, population, verbose=False):
    for ind_i, (score, ind) in enumerate(population):
        for hof_j, (hof_score, hof_ind) in enumerate(hof):
            if hof_score is None:
                hof.insert(hof_j, (score, ind))
                if verbose: print(f'Replacing HOF individual {hof_j}, new score of {score}')
                hof.pop()
                break

            if score < hof_score:
                hof.insert(hof_j, (score, ind))
                if verbose: print(f'Replacing HOF individual {hof_j}, new score of {score}')
                hof.pop()
                break
    return hof


def topology_random_search(graph, propagator, evaluator, evolver, io, ga_opts):
    hof = init_hof(ga_opts['n_hof'])
    log, log_metrics = logbook_initialize()
    save_all_graphs = False

    io.save_json(ga_opts, 'ga_opts.json')
    for (object_filename, object_to_save) in zip(('propagator', 'evaluator', 'evolver'), (propagator, evaluator, evolver)):
        io.save_object(object_to_save=object_to_save, filename=f"{object_filename}.pkl")

    _, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    if ga_opts['multiprocess']:
        pool = mp.Pool(mp.cpu_count())
        print('Starting multiprocessing with {} CPUs'.format(mp.cpu_count()))

    population = []
    for individual in range(ga_opts['n_population']):
        population.append((None, copy.deepcopy(graph)))

    for generation in range(ga_opts['n_generations']):
        print('Generation {}'.format(generation))

        # pass in a list, each element is a list, consisting of
        args = [(evolver, graph, evaluator, propagator) for (score, graph) in population]
        if ga_opts['multiprocess']:
            population = pool.map(parameters_optimize_multiprocess, args)
        else:
            population = list(map(parameters_optimize_multiprocess, args))

        population.sort(reverse = False, key=lambda x: x[0])  # we sort ascending, and take first (this is the minimum, as we minimizing)

        # save all graphs if desired
        if save_all_graphs:
            for i, (score, graph) in enumerate(population):
                graph.score = score
                graph.clear_propagation()
                io.save_object(object_to_save=graph, filename=f"graph_gen{generation}_{i}.pkl")

        logbook_update(generation, population, log, log_metrics, verbose=ga_opts['verbose'])

        hof = update_hof(hof=hof, population=population, verbose=ga_opts['verbose'])

    fig, axs = plt.subplots(nrows=ga_opts['n_hof'], ncols=2, figsize=[5, 2*ga_opts['n_hof']])
    for i, (score, graph) in enumerate(hof):
        graph.score = score
        graph.clear_propagation()
        io.save_object(object_to_save=graph, filename=f"graph_hof{i}.pkl")

        state = graph.measure_propagator(-1)
        graph.draw(ax=axs[i,0], legend=True)
        axs[i,1].plot(propagator.t, evaluator.target, label='Target')
        axs[i,1].plot(propagator.t, np.power(np.abs(state), 2), label='Solution')

    io.save_fig(fig=fig, filename='halloffame.pdf')

    return hof[1], hof[0], log


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