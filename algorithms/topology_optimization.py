"""
Random search for topology for benchmarking
"""
import time
import autograd.numpy as np
import matplotlib.pyplot as plt
import copy
import ray
import random
from autograd.numpy.numpy_boxes import ArrayBox

import networkx as nx
import config.config as config

from algorithms.functions import logbook_update, logbook_initialize
from .parameter_optimization import parameters_optimize
from algorithms.speciation import Speciation, NoSpeciation, SimpleSubpopulationSchemeDist, vectorDIFF, photoNEAT
import config.config as configuration

SPECIATION_MANAGER = NoSpeciation() 

def topology_optimization(graph, propagator, evaluator, evolver, io,
                          crossover_maker=None, parameter_opt_method='L-BFGS+GA',
                          ga_opts=None, update_rule='random',
                          target_species_num=4, protection_half_life=None,
                          cluster_address=None, local_mode=False, include_dashboard=False):
    io.init_logging()
    log, log_metrics = logbook_initialize()
    # # prev test we were using
    # random.seed(18)
    # np.random.seed(1040)

    # new test
    # random.seed(15)
    # np.random.seed(10480)

    if update_rule == 'random':
        update_population = update_population_topology_random  # set which update rule to use
    elif update_rule == 'preferential':
        update_population = update_population_topology_preferential
    elif update_rule == 'roulette':
        update_population = update_population_topology_roulette
    elif update_rule == 'tournament':
        update_population = update_population_topology_tournament
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
    # ray.init(address=cluster_address, num_cpus=ga_opts['num_cpus'], local_mode=local_mode, ignore_reinit_error=True) #, object_store_memory=1e9)
    ray.init(address=cluster_address, num_cpus=ga_opts['num_cpus'], local_mode=local_mode, include_dashboard=include_dashboard, ignore_reinit_error=True) #, object_store_memory=1e9)
    evaluator_id, propagator_id = ray.put(evaluator), ray.put(propagator)

    # start_graph = ray.put(copy.deepcopy(graph))


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
    score, graph = parameters_optimize_complete((None, graph), evaluator, propagator, method=parameter_opt_method)
    graph.score = score
    for individual in range(ga_opts['n_population']):
        population.append((score, copy.deepcopy(graph)))

    t1 = time.time()
    for generation in range(ga_opts['n_generations']):
        print(f'\ngeneration {generation} of {ga_opts["n_generations"]}: time elapsed {time.time()-t1}s')

        population = update_population(population, evolver, evaluator, target_species_num=target_species_num, # ga_opts['n_population'] / 20,
                                                                       protection_half_life=protection_half_life,
                                                                       crossover_maker=crossover_maker)
        print(f'population length after update: {len(population)}')
        
        # optimize parameters on each node/CPU
        population = ray.get([parameters_optimize_multiprocess.remote(ind,
                                                                      evaluator_id,
                                                                      propagator_id,
                                                                      method=parameter_opt_method,
                                                                      verbose=ga_opts['verbose']) for ind in population])
        save_scores_to_graph(population) # necessary for some algorithms
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

    io.save_object(log, 'log.pkl')

    io.close_logging()
    evolver.close()
    return hof, log # hof is actually a list in and of itself, so we only look at the top element

def save_hof(hof, io):
    for i, (score, graph) in enumerate(hof):
        io.save_object(object_to_save=graph.duplicate_and_simplify_graph(graph), filename=f"graph_hof{i}.pkl")
    return

def plot_hof(hof, propagator, evaluator, io):
    # save a figure which quickly demonstrates the results of the run as a .pdf files
    fig, axs = plt.subplots(nrows=len(hof), ncols=3, figsize=[5, 2 * len(hof)])
    for i, (score, graph) in enumerate(hof):
        graph.score = score
        graph.propagate(propagator, save_transforms=False)

        measurement_node = 'sink'
        state = graph.measure_propagator(measurement_node)
        if len(hof) > 1:
            np.expand_dims(axs, 0)

        graph.draw(ax=axs[i, 0], debug=False)
        axs[i, 1].plot(propagator.t, evaluator.target, label='Target')
        axs[i, 1].plot(propagator.t, np.power(np.abs(state), 2), label='Solution')
        axs[i, 2].set(xticks=[], yticks=[])
        axs[i, 2].grid = False
        axs[i, 2].annotate("Score:\n{:2.3e}".format(score), xy=[0.5, 0.5], xycoords='axes fraction', va='center', ha='center')
    io.save_fig(fig=fig, filename='halloffame.png')

def save_scores_to_graph(population):
    for (score, graph) in population:
        graph.score = score


def init_hof(n_hof):
    hof = [(None, None) for i in range(n_hof)]
    return hof

def graph_kernel_map_to_nodetypes(_graph):
    """
    NOT SUPPORTED AFTER GRAPH ENCODING CHANGE.
    A pre-processing step to collapse nodes to their model types.
    :param _graph:
    :return:
    """
    graph_relabelled = nx.relabel_nodes(_graph, {node: _graph.nodes[node]['model'].node_acronym for node in _graph.nodes})
    all_node_relabels = []
    for node_subtypes in config.NODE_TYPES_ALL.values():
        for node_subtype in node_subtypes.values():
            all_node_relabels.append(node_subtype.node_acronym)
    graph_relabelled.add_nodes_from(all_node_relabels)
    return graph_relabelled

def similarity_full_ged(g1, g2):
    """
    Measures the Graph Edit Distance similarity between two graphs exactly. Can be slow, it is suggested use the
    approximate (reduced) method instead
    :param _graph1: Graph object
    :param _graph2: Graph object
    :return: similarity (integer number of steps needed to transform Graph 1 to Graph 2
    """
    sim = nx.algorithms.similarity.graph_edit_distance(g1, g2,
                                                       edge_subst_cost=edge_node_match,
                                                       node_subst_cost=edge_node_match,
                                                       upper_bound=30.0,
                                                       timeout=10.0,
                                                       )
    return sim

def similarity_reduced_ged(g1, g2):
    """
    Approximated the Graph Edit Distance similarity between two graphs exactly.
    :param _graph1: Graph object
    :param _graph2: Graph object
    :return: similarity (integer number of steps needed to transform Graph 1 to Graph 2
    """
    ged_approx = nx.algorithms.similarity.optimize_graph_edit_distance(g1, g2,
                                                                       edge_subst_cost=edge_node_match,
                                                                       node_subst_cost=edge_node_match,
                                                                       upper_bound=30.0,
                                                                       )
    sim =  next(ged_approx) # faster, but less accurate
    return sim

def edge_node_match(e1, e2):
    # provides the comparison for the cost of substituting two edges or two nodes in the GED calculation
    if type(e1['model']) == type(e2['model']):
        cost = 0.0
    else:
        cost = 1.0
    return cost

def update_hof(hof, population, similarity_measure='reduced_ged', threshold_value=0.0, verbose=False):
    """
    :param hof: list of N tuples, where each tuple is (score, graph) and are the best performing of the entire run so far
    :param population: current population of graphs, list of M tuples, with each tuple (score, graph)
    :param similarity_measure: string identifier for which similarity/distance measure to use. currently implemented are
        'reduced_ged': a graph reduction method before using the Graph Edit Distance measurement,
        'full_ged': directly using the Graph Edit Distance measurement on the system graphs
    :param threshold_value: positive float. two graphs with similarity below this value are considered to be the same structure
    :param verbose: debugging to print, boolean
    :return: returns the updated hof, list of tuples same as input
    """

    if similarity_measure == 'reduced_ged':
        similarity_function = similarity_reduced_ged
    elif similarity_measure == 'full_ged':
        similarity_function = similarity_full_ged
    else:
        raise NotImplementedError('this is not an implemented graph measure function. please use reduced_ged or full_ged.')


    for i, (score, graph) in enumerate(population):
        if verbose: print(f'\n\nNow checking with population index {i}')

        insert = False
        insert_ind = None
        remove_ind = -1
        check_similarity = True
        for j, (hof_j_score, hof_j) in enumerate(hof):
            if verbose: print(f'checking score against index {j} of the hof')
            if hof_j_score is None:
                insert = True
                insert_ind = j
                check_similarity = True
                break

            # if performing better, go to the next hof candidate with the next best score
            elif score < hof_j_score:
                insert = True
                insert_ind = j
                check_similarity = True
                if verbose: print(f'Better score than index {j} of the hof')
                break

            else:
                # no need to check similarity if the score is worse than all hof graphs
                check_similarity = False

        if not check_similarity:
            if verbose: print(f'There is no need to check the similarity')


        if check_similarity and (similarity_measure is not None):
            # similarity check with all HoF graphs
            for k, (hof_k_score, hof_k) in enumerate(hof):
                if verbose: print(f'Comparing similarity of with population index {i} with hof index {k}')

                if hof_k is not None:
                    sim = similarity_function(graph, hof_k)

                    if sim < threshold_value:
                        # there is another, highly similar graph in the hof
                        if k < j:
                            # there is a similar graph with a better score, do not add graph to hof
                            insert = False
                            if verbose: print(f'A similar, but better performing graph exists - the current graph will not be added')
                            break # breaks out of 'k' loop
                        elif k >= j:
                            # there is a similar graph with a worse score, add in that location instead
                            insert = True
                            remove_ind = k
                            if verbose: print(f'A similar graph exists at index {k} of the hof. The current graph scores better and will be added')
                            break
                    else:
                        if verbose: print(f'Similarity of {sim} is not below the threshold')
                        pass
        if insert: # places this graph into the insert_ind index in the halloffame
            hof[remove_ind] = 'x'
            hof.insert(insert_ind, (score, copy.deepcopy(graph)))
            hof.remove('x')
            if verbose: print(f'Replacing HOF individual {remove_ind}, new score of {score}')
        else:
            if verbose: print(f'Not adding population index {i} into the hof')


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
                print(f'Could not evolve this graph.{[node for node in graph_tmp.nodes()]}\n {graph}')
                continue

            if len(x0) == 0:
                continue
            else:
                graph = graph_tmp
                break
        population[i] = (None, graph)

    return population


def update_population_topology_preferential(population, evolver, evaluator, preferentiality_param=2, **hyperparameters):
    """
    Updates population such that the fitter individuals have a larger chance of reproducing

    So we want individuals to reproduce about once on average, more for the fitter, less for the less fit
    Does not modify the original graphs in population, unlike the random update rule

    :pre-condition: population is sorted in ascending order of score (i.e. most fit to least fit)
    """
    most_fit_reproduction_mean = 2 # most fit element will on average reproduce this many additional times (everyone reproduces once at least)
    
    # 1. Initialize scores (only happens on generation 0)
    if (population[0][0] is not None):
        score_array = np.array([score for (score, _) in population])
    else:
        score_array = np.ones(len(population)).reshape(len(population), 1)
        most_fit_reproduction_mean = 1
    
    new_pop = []

    # 2. Mutate existing elements (fitter individuals have a higher expectation value for number of reproductions)
    # basically after the initial reproduction, the most fit reproduces on average <most_fit_reproduction_mean> times, and the least fit never reproduces
    
    # TODO: test different ways of getting this probability (one that favours the top individuals less?)
    break_probability = 1 / most_fit_reproduction_mean * (score_array / np.min(score_array))**(1 / preferentiality_param)

    print(f'break prob: {break_probability}')

    for i, (score, graph) in enumerate(population):
        x, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
        parent_parameters = copy.deepcopy(x)
        while True:
            graph_tmp, evo_op_choice = evolver.evolve_graph(copy.deepcopy(graph), evaluator)
            try:
                graph.distribute_parameters_from_list(parent_parameters, node_edge_index, parameter_index) # This is a hacky fix because smh graph parameters are occasionally modified through the deepcopy
            except IndexError as e:
                print(e)
                print(graph)
                print(f'parent parameters: {parent_parameters}')
                raise e

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

    from_last_gen = population[0:len(population) // 10 + 1]

    return from_last_gen + new_pop # top 10 percent of old population, and new recruits go through


def update_population_topology_roulette(population, evolver, evaluator, preferentiality_param=1, **hyperparameters):
    """
    Updates population such that the fitter individuals have a larger chance of reproducing, using the "roulette wheel" method

    So we want individuals to reproduce about once on average, more for the fitter, less for the less fit
    Does not modify the original graphs in population, unlike the random update rule

    :pre-condition: population is sorted in ascending order of score (i.e. most fit to least fit)
    :param preferentiality_param: the larger the parameter, the more fitter individuals (lower score) are favoured
    """    
    # 1. Initialize scores (only happens on generation 0)
    if (population[0][0] is not None):
        score_array = np.array([score for (score, _) in population])
    else:
        score_array = np.ones(len(population)).reshape(len(population), 1)
    
    new_pop = []

    # 2. Build roulette
    probability_roulette = 1 / score_array**preferentiality_param
    probability_roulette /= np.sum(probability_roulette)
    print(f'scores: {score_array}')
    print(f'probablities: {probability_roulette}')

    # 3. Sample using roulette
    for _ in range(len(population)):
        reproduction_index = np.random.choice(np.arange(0, len(population)), p=probability_roulette)
        graph = population[reproduction_index][1]
        graph_tmp, _ = evolver.evolve_graph(copy.deepcopy(graph), evaluator)
        graph_tmp.assert_number_of_edges()
        # TODO: determine whether the following is a correct check to make (or is it just alright?)
        # x0, node_edge_index, parameter_index, *_ = graph_tmp.extract_parameters_to_list()
        # if (len(x0)) == 0:
        #     raise ValueError('This graph has no parameters')
        new_pop.append((None, graph_tmp))
    
    return new_pop


def update_population_topology_tournament(population, evolver, evaluator, tournament_size_divisor=3, **hyperparameters):
    """
    Updates population such that the fitter individuals have a larger chance of reproducing, using the "roulette wheel" method

    So we want individuals to reproduce about once on average, more for the fitter, less for the less fit
    Does not modify the original graphs in population, unlike the random update rule

    :pre-condition: population is sorted in ascending order of score (i.e. most fit to least fit)
    :param tournament_size_divisor: tournament size as ratio of total population size
                                    (e.g. for population size = 9, tournament_size_divisor=3, tournament size = 9 / 3 = 3)
    """    
    # 1. Initialize scores (only happens on generation 0)
    if (population[0][0] is not None):
        score_array = np.array([score for (score, _) in population])
    else:
        score_array = np.ones(len(population)).reshape(len(population), 1)
    
    new_pop = []

    # 2. Setup tournament
    tournament_size = len(population) // tournament_size_divisor
    if tournament_size == 0:
        print(f'WARNING: tournament size is zero. Setting size to 1')
        tournament_size = 1

    # 3. Sample using tournament
    for _ in range(len(population)):
        tournament_indices = np.random.choice(np.arange(0, len(population)), size=tournament_size)
        graph = population[min(tournament_indices)][1] # this works, because the vals were already sorted by fitness
        print(f'tournament indices: {tournament_indices}')
        print(f'graph index: {min(tournament_indices)}')
        graph_tmp, _ = evolver.evolve_graph(copy.deepcopy(graph), evaluator)
        graph_tmp.assert_number_of_edges()
        # TODO: determine whether the following is a correct check to make (or is it just alright?)
        # x0, node_edge_index, parameter_index, *_ = graph_tmp.extract_parameters_to_list()
        # if (len(x0)) == 0:
        #     raise ValueError('This graph has no parameters')
        new_pop.append((None, graph_tmp))
    
    return new_pop
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


def parameters_optimize_complete(ind, evaluator, propagator, method='NULL', verbose=True):
    score, graph = ind
    if score is not None:
        return score, graph

    try:
        graph.update_graph()  # updates propaation order and input/outputs on nodes
        graph.clear_propagation()
        graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
        x0, model, parameter_index, *_ = graph.extract_parameters_to_list()
        graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
        graph, parameters, score, log = parameters_optimize(graph, x0=x0, method=method, verbose=verbose)
        return score, graph
    except Exception as e:
        print(f'error caught in parameter optimization: {e}')
        raise e
        return 99999999, graph


@ray.remote
def parameters_optimize_multiprocess(ind, evaluator, propagator, method='NULL', verbose=True):
    return parameters_optimize_complete(ind, evaluator, propagator, method=method, verbose=verbose)