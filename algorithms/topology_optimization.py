"""
Random search for topology for benchmarking
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import ray
import random

from algorithms.functions import logbook_update, logbook_initialize
from .parameter_optimization import parameters_optimize
from algorithms.speciation import Speciation, NoSpeciation, SimpleSubpopulationSchemeDist, vectorDIFF, photoNEAT
import config.config as configuration

SPECIATION_MANAGER = NoSpeciation() 

def topology_optimization(graph, propagator, evaluator, evolver, io,
                          crossover_maker=None,
                          ga_opts=None, update_rule='random',
                          target_species_num=4, protection_half_life=None,
                          cluster_address=None, local_mode=False):
    io.init_logging()
    log, log_metrics = logbook_initialize()
    random.seed(18)
    np.random.seed(1040)

    if update_rule == 'random':
        update_population = update_population_topology_random  # set which update rule to use
    elif update_rule == 'preferential':
        update_population = update_population_topology_preferential
    elif update_rule == 'random simple subpop scheme':
        update_population = update_population_topology_random_simple_subpopulation_scheme
    elif update_rule == 'preferential simple subpop scheme':
        update_population = update_population_topology_preferential_simple_subpopulation_scheme
    elif update_rule == 'random vectorDIFF':
        update_population = update_population_topology_random_vectorDIFF
    elif update_rule == 'preferential vectorDIFF':
        update_population = update_population_topology_preferential_vectorDIFF
    elif update_rule == 'random photoNEAT':
        update_population = update_population_topology_random_photoNEAT
    elif update_rule == 'preferential photoNEAT':
        update_population = update_population_topology_preferential_photoNEAT
    else:
        raise NotImplementedError("This topology optimization update rule is not implemented yet. current options are 'random'")

    # start up the multiprocessing/distributed processing with ray, and make objects available to nodes
    if local_mode: print(f"Running in local_mode - not running as distributed computation")
    ray.init(address=cluster_address, num_cpus=ga_opts['num_cpus'], local_mode=local_mode, include_dashboard=False, ignore_reinit_error=True)
    # ray.init(address=cluster_address, num_cpus=ga_opts['num_cpus'], local_mode=local_mode, ignore_reinit_error=True) #, object_store_memory=1e9)
    evaluator_id, propagator_id = ray.put(evaluator), ray.put(propagator)

    # save the objects for analysis later
    io.save_json(ga_opts, 'ga_opts.json')
    for (object_filename, object_to_save) in zip(('propagator', 'evaluator', 'evolver'), (propagator, evaluator, evolver)):
        io.save_object(object_to_save=object_to_save, filename=f"{object_filename}.pkl")
    
    #save optimization settings for analysis later
    optimization_settings = {'update_rule':update_rule, 'target_species_num':target_species_num, 'protection_half_life':protection_half_life}
    io.save_json(optimization_settings, 'optimization_settings.json')

    # create initial population and hof
    hof = init_hof(ga_opts['n_hof'])
    population = []
    for individual in range(ga_opts['n_population']):
        population.append((None, copy.deepcopy(graph)))

    t1 = time.time()
    for generation in range(ga_opts['n_generations']):
        print(f'\ngeneration {generation} of {ga_opts["n_generations"]}: time elapsed {time.time()-t1}s')

        population = update_population(population, evolver, evaluator, target_species_num=target_species_num, # ga_opts['n_population'] / 20,
                                                                       protection_half_life=protection_half_life,
                                                                       crossover_maker=crossover_maker)
        print(f'population length after update: {len(population)}')
        
        # optimize parameters on each node/CPU
        population = ray.get([parameters_optimize_multiprocess.remote(ind, evaluator_id, propagator_id) for ind in population])
        hof = update_hof(hof=hof, population=population, verbose=ga_opts['verbose']) # update before speciation, since we don't want this hof score affected by speciation
        SPECIATION_MANAGER.speciate(population)
        SPECIATION_MANAGER.execute_fitness_sharing(population, generation)

        population.sort(reverse = False, key=lambda x: x[0])  # we sort ascending, and take first (this is the minimum, as we minimizing)
        population = population[0:ga_opts['n_population']] # get rid of extra params, if we have too many
        SPECIATION_MANAGER.reverse_fitness_sharing(population, generation) # ensures log info is correct (fitness sharing is ONLY to select next gen)
        for (score, graph) in population:
            graph.clear_propagation()

        # update logbook and hall of fame
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
        if ga_opts['n_hof'] > 1:
            graph.draw(ax=axs[i,0], legend=True)
            axs[i,1].plot(propagator.t, evaluator.target, label='Target')
            axs[i,1].plot(propagator.t, np.power(np.abs(state), 2), label='Solution')
        else:
            graph.draw(ax=axs[0], legend=True)
            axs[1].plot(propagator.t, evaluator.target, label='Target')
            axs[1].plot(propagator.t, np.power(np.abs(state), 2), label='Solution')

    io.save_fig(fig=fig, filename='halloffame.png')
    io.save_object(log, 'log.pkl')

    io.close_logging()

    return hof[0][1], hof[0][0], log # hof is actually a list in and of itself, so we only look at the top element

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


def update_population_topology_random(population, evolver, evaluator, **hyperparameters):
    # mutating the population occurs on head node, then graphs are distributed to nodes for parameter optimization
    for i, (score, graph) in enumerate(population):
        while True:
            graph_tmp, evo_op_choice = evolver.evolve_graph(graph, evaluator)
            x0, node_edge_index, parameter_index, *_ = graph_tmp.extract_parameters_to_list()
            try:
                graph_tmp.assert_number_of_edges()
            except:
                print(f'Could not evolve this graph.{[node for node in graph_tmp.nodes()]}')
                continue

            if len(x0) == 0:
                continue
            else:
                graph = graph_tmp
                break
        population[i] = (None, graph)

    return population


# def update_population_crossover_test(population, evolver, evaluator, **hyperparameters):
#     graph0 = population[0][1]
#     graph1 = population[1][1]
#     graph0.draw(legend=True)
#     # graph1.draw(legend=True)
#     # x00, node_edge_index0, parameter_index0, *_ = graph0.extract_parameters_to_list()
#     # x01, node_edge_index1, parameter_index1, *_ = graph1.extract_parameters_to_list()
#     # print(f'x0, node_edge_index, parameter_index (graph0, graph1): {(x00, x01)}, {(node_edge_index0, node_edge_index1)}, {(parameter_index0, parameter_index1)}')



#     verification = [evo_op().verify_evolution(graph0, graph1) for (_, evo_op) in configuration.CROSSOVER_OPERATORS.items()]
#     if not verification[0]:
#         print(f'verification failed!')
#         assert False
#     possible_crossover_ops = [evo_op for (verify, evo_op) in zip(verification, configuration.CROSSOVER_OPERATORS.values()) if verify]
    

#     crossover = possible_crossover_ops[0]()

#     child0, child1 = crossover.apply_evolution(graph0, graph1)

#     # child0.draw(legend=True)
#     # child1.draw(legend=True)
#     # plt.show()
#     population[0] = (None, child0)
#     population[1] = (None, child1)

#     return population

def update_population_topology_preferential(population, evolver, evaluator, preferentiality_param=2, **hyperparameters):
    """
    Updates population such that the fitter individuals have a larger chance of reproducing

    So we want individuals to reproduce about once on average, more for the fitter, less for the less fit
    Does not modify the original graphs in population, unlike the random update rule

    :pre-condition: population is sorted in ascending order of score (i.e. most fit to least fit)
    """
    print(f'update population input population size: {len(population)}')
    most_fit_reproduction_mean = 2 # most fit element will on average reproduce this many additional times (everyone reproduces once at least)
    
    # 1. Initialize scores (only happens on generation 0)
    if (population[0][0] is not None):
        score_array = np.array([score for (score, _) in population])
    else:
        score_array = np.ones(len(population)).reshape(len(population), 1)
        most_fit_reproduction_mean = 1
    
    new_pop = []
    # 2. Execute crossovers, if crossovers enabled
    if hyperparameters['crossover_maker'] is not None:
        # top 10% reproduce, reproduces with fitter mate with higher probability
        will_reproduce = population[0:len(population) // 10 + 1]
        could_reproduce = set(population)
        for ind in will_reproduce:
            if ind in could_reproduce:
                could_reproduce.remove(ind)
                compatible_mates = SPECIATION_MANAGER.get_crossover_candidates(ind, could_reproduce) # note: there's never reproduction on round 1, would be useless
                if len(compatible_mates) == 0:
                    continue
                reproduction_prob = np.array([2**(len(compatible_mates) - i - 1) for i in range(len(compatible_mates))])
                reproduction_prob = reproduction_prob / np.sum(reproduction_prob)
                mate_indices = np.arange(0, len(compatible_mates))
                mate = compatible_mates[np.random.choice(mate_indices, p=reproduction_prob)] # returns individual (score, Graph) not just graph

                child0, child1, _ = hyperparameters['crossover_maker'].crossover_graphs(copy.deepcopy(ind[1]), copy.deepcopy(mate[1]))
                new_pop.append((None, child0))
                new_pop.append((None, child1))
                could_reproduce.remove(mate)

    # 3. Mutate existing elements (fitter individuals have a higher expectation value for number of reproductions)
    # basically after the initial reproduction, the most fit reproduces on average <most_fit_reproduction_mean> times, and the least fit never reproduces
    
    # TODO: test different ways of getting this probability (one that favours the top individuals less?)
    break_probability = 1 / most_fit_reproduction_mean * (score_array / np.min(score_array))**(1 / preferentiality_param)

    print(f'break prob: {break_probability}')
    for i, (score, graph) in enumerate(population):
        while True:
            graph_tmp, evo_op_choice = evolver.evolve_graph(copy.deepcopy(graph), evaluator)
            x0, node_edge_index, parameter_index, *_ = graph_tmp.extract_parameters_to_list()
            try:
                graph_tmp.assert_number_of_edges()
            except:
                continue

            if len(x0) == 0:
                continue
            
            new_pop.append((None, graph_tmp))
            if random.random() < break_probability[i]:
                break
    # speciation manager reverses fitness sharing to get raw score. Generation - 1 is given, since we're reversing the sharing of the previous generation in this case
    from_last_gen = population[0:len(population) // 10 + 1]
    # print(f'from last gen before fitness reversal: {from_last_gen}')
    # if population[0][0] is not None:
    #     SPECIATION_MANAGER.reverse_fitness_sharing(from_last_gen, hyperparameters['generation_num'] - 1)
    # print(f'from last gen after fitness reversal: {from_last_gen}')
    return from_last_gen + new_pop # top 10 percent of old population, and new recruits go through


# ------------------------------- Speciation setup helpers -----------------------------------

def _simple_subpopulation_setup(population, **hyperparameters):
    global SPECIATION_MANAGER
    # set up speciation if it's not set up already. This is only necessary on first function call
    if population[0][1].speciation_descriptor is None:
        dist_func = SimpleSubpopulationSchemeDist().distance
        SPECIATION_MANAGER = Speciation(target_species_num=hyperparameters['target_species_num'],
                                        protection_half_life=hyperparameters['protection_half_life'],
                                        distance_func=dist_func)
        for i, (_, graph) in enumerate(population):
            graph.speciation_descriptor = {'name':'simple subpopulation scheme', 'label':i % hyperparameters['target_species_num']}


def _vectorDIFF_setup(population, **hyperparameters):
    global SPECIATION_MANAGER
    # set up speciation if it's not set up already, noly necessary on first function call
    if population[0][1].speciation_descriptor is None:
        dist_func = vectorDIFF().distance
        SPECIATION_MANAGER = Speciation(target_species_num=hyperparameters['target_species_num'],
                                        protection_half_life=hyperparameters['protection_half_life'],
                                        distance_func=dist_func)
        for _, graph in population:
            graph.speciation_descriptor = {'name':'vectorDIFF'}


def _photoNEAT_setup(population, **hyperparameters):
    global SPECIATION_MANAGER
    # set up speciation if it's not set up already, only necessary on first function call
    if population[0][1].speciation_descriptor is None:
        dist_func = photoNEAT().distance
        SPECIATION_MANAGER = Speciation(target_species_num=hyperparameters['target_species_num'],
                                        protection_half_life=hyperparameters['protection_half_life'],
                                        distance_func=dist_func)
        for i, (_, graph) in enumerate(population):
            marker_node_map = {}
            node_marker_map = {}
            for i, node in enumerate(graph.nodes): # we only expect 2 things to start with, so i = 0, 1
                marker_node_map[i] = node
                node_marker_map[node] = i
            graph.speciation_descriptor = {'name':'photoNEAT', 'marker to node':marker_node_map, 'node to marker':node_marker_map}


# ---------------------------------- Speciated population update ------------------------------
def update_population_topology_random_simple_subpopulation_scheme(population, evolver, evaluator, **hyperparameters):
    _simple_subpopulation_setup(population, **hyperparameters)
    return update_population_topology_random(population, evolver, evaluator, **hyperparameters) # rest goes on normally


def update_population_topology_preferential_simple_subpopulation_scheme(population, evolver, evaluator, **hyperparameters):
    _simple_subpopulation_setup(population, **hyperparameters)
    return update_population_topology_preferential(population, evolver, evaluator, **hyperparameters) # rest goes on normally


def update_population_topology_random_vectorDIFF(population, evolver, evaluator, **hyperparameters):
    _vectorDIFF_setup(population, **hyperparameters)
    return update_population_topology_random(population, evolver, evaluator, **hyperparameters)


def update_population_topology_preferential_vectorDIFF(population, evolver, evaluator, **hyperparameters):
    _vectorDIFF_setup(population, **hyperparameters)
    return update_population_topology_preferential(population, evolver, evaluator, **hyperparameters)


def update_population_topology_random_photoNEAT(population, evolver, evaluator, **hyperparameters):
    """
    NOTE: with random population updates, NEAT speciation will not be very useful! That's because each individual just
    mutates once and moves on, which means that the speciation will brutally branch out and never recover

    Would be more useful with crossovers, or a number of offspring proportional which depends on their fitness
    """
    _photoNEAT_setup(population, **hyperparameters)
    return update_population_topology_random(population, evolver, evaluator, **hyperparameters)


def update_population_topology_preferential_photoNEAT(population, evolver, evaluator, **hyperparameters):
    _photoNEAT_setup(population, **hyperparameters)
    return update_population_topology_preferential(population, evolver, evaluator, **hyperparameters)


def parameters_optimize_complete(ind, evaluator, propagator):
    score, graph = ind
    if score is not None:
        return score, graph

    try:
        graph.clear_propagation()
        graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
        x0, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
        graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
        graph, parameters, score, log = parameters_optimize(graph, x0=x0, method='L-BFGS+GA', verbose=False)

        return score, graph
    except Exception as e:
        print(f'error caught in parameter optimization: {e}')
        return 99999999, graph


@ray.remote
def parameters_optimize_multiprocess(ind, evaluator, propagator):
    return parameters_optimize_complete(ind, evaluator, propagator)