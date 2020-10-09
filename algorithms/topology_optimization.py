"""
Random search for topology for benchmarking
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import ray

from algorithms.functions import logbook_update, logbook_initialize
from .parameter_optimization import parameters_optimize
from algorithms.speciation import Speciation, SimpleSubpopulationSchemeDist


SPECIATION_MANAGER = None 


def topology_optimization(graph, propagator, evaluator, evolver, io,
                          ga_opts=None, update_rule='random',
                          cluster_address=None, local_mode=False):
    io.init_logging()
    log, log_metrics = logbook_initialize()

    if update_rule == 'random':
        update_population = update_population_topology_random  # set which update rule to use
    elif update_rule == 'random simple subpop scheme':
        update_population = update_population_topology_random_simple_subpopulation_scheme
    else:
        raise NotImplementedError("This topology optimization update rule is not implemented yet. current options are 'random'")

    # start up the multiprocessing/distributed processing with ray, and make objects available to nodes
    if local_mode: print(f"Running in local_mode - not running as distributed computation")
    # ray.init(address=cluster_address, num_cpus=ga_opts['num_cpus'], local_mode=local_mode, include_dashboard=False, ignore_reinit_error=True)
    ray.init()
    evaluator_id, propagator_id = ray.put(evaluator), ray.put(propagator)

    # save the objects for analysis later
    io.save_json(ga_opts, 'ga_opts.json')
    for (object_filename, object_to_save) in zip(('propagator', 'evaluator', 'evolver'), (propagator, evaluator, evolver)):
        io.save_object(object_to_save=object_to_save, filename=f"{object_filename}.pkl")

    # create initial population and hof
    hof = init_hof(ga_opts['n_hof'])
    population = []
    for individual in range(ga_opts['n_population']):
        population.append((None, copy.deepcopy(graph)))

    t1 = time.time()
    for generation in range(ga_opts['n_generations']):
        print(f'\ngeneration {generation} of {ga_opts["n_generations"]}: time elapsed {time.time()-t1}s')

        population = update_population(population, evolver, evaluator, propagator, target_species_num=ga_opts['n_population'] / 20,
                                                                                   protection_half_life=None)

        # optimize parameters on each node/CPU
        population = ray.get([parameters_optimize_multiprocess.remote(graph, evaluator_id, propagator_id) for (_, graph) in population])
        if SPECIATION_MANAGER is not None: # apply fitness sharing
            SPECIATION_MANAGER.speciate(population)
            SPECIATION_MANAGER.execute_fitness_sharing(population, generation)

        population.sort(reverse = False, key=lambda x: x[0])  # we sort ascending, and take first (this is the minimum, as we minimizing)
        for (score, graph) in population:
            graph.clear_propagation()

        # update logbook and hall of fame
        hof = update_hof(hof=hof, population=population, verbose=ga_opts['verbose'])
        logbook_update(generation, population, log, log_metrics, time=(time.time()-t1), best=hof[0][0], verbose=ga_opts['verbose'])

    # save a figure which quickly demonstrates the results of the run as a .pdf files
    fig, axs = plt.subplots(nrows=ga_opts['n_hof'], ncols=2, figsize=[5, 2*ga_opts['n_hof']])
    for i, (score, graph) in enumerate(hof):
        graph.score = score

        graph.propagate(propagator, save_transforms=False)
        hof_i_score = evaluator.evaluate_graph(graph, propagator)
        if score != hof_i_score: print("HOF final score calculation does not match")

        io.save_object(object_to_save=graph, filename=f"graph_hof{i}.pkl")

        state = graph.measure_propagator(-1)
        graph.draw(ax=axs[i,0], legend=True)
        axs[i,1].plot(propagator.t, evaluator.target, label='Target')
        axs[i,1].plot(propagator.t, np.power(np.abs(state), 2), label='Solution')

    io.save_fig(fig=fig, filename='halloffame.png')
    io.save_object(log, 'log.pkl')

    io.close_logging()
    return hof[1], hof[0], log

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


def update_population_topology_random(population, evolver, evaluator, propagator, **hyperparameters):
    # mutating the population occurs on head node, then graphs are distributed to nodes for parameter optimization
    for i, (score, graph) in enumerate(population):
        while True:
            graph_tmp, evo_op_choice = evolver.evolve_graph(graph, evaluator, propagator)
            x0, node_edge_index, parameter_index, *_ = graph_tmp.extract_parameters_to_list()
            try:
                graph_tmp.assert_number_of_edges()
            except:
                continue

            if len(x0) == 0:
                continue
            else:
                graph = graph_tmp
                break
        population[i] = (None, graph)
    return population


def update_population_topology_random_simple_subpopulation_scheme(population, evolver, evaluator, propagator, **hyperparameters):
    global SPECIATION_MANAGER
    # set up speciation if it's not set up already. This is only necessary on first function call
    if population[0][1].speciation_descriptor is None:
        dist_func = SimpleSubpopulationSchemeDist(num_species=hyperparameters['target_species_num']).distance
        SPECIATION_MANAGER = Speciation(target_species_num=hyperparameters['target_species_num'],
                                        protection_half_life=hyperparameters['protection_half_life'],
                                        distance_func=dist_func)
        for i, (_, graph) in enumerate(population()):
            graph.speciation_descriptor = i % hyperparameters['target_species_num']

    return update_population_topology_random(population, evolver, evaluator, propagator) # rest goes on normally
    


@ray.remote
def parameters_optimize_multiprocess(graph, evaluator, propagator):
    graph.clear_propagation()
    graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
    graph, parameters, score, log = parameters_optimize(graph, x0=x0, method='L-BFGS+GA', verbose=False)
    return score, graph