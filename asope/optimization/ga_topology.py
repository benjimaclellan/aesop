
#%% import public libraries
import numpy as np
from deap import tools, base, creator, algorithms
import multiprocess as mp
import copy
import matplotlib.pyplot as plt

#%% import custom modules
from optimization.geneticalgorithm import eaSimple, initialgeneration, newchildren
from assets.graph_manipulation import change_experiment_wrapper, brand_new_experiment, remake_experiment
from classes.experiment import Experiment
from optimization.wrappers import optimize_experiment
from assets.functions import splitindices

#TODO: ALl functions for topology manipulations can/should be improved

# %%
"""
Creates a new individual for topology optimization (i.e. a new topology)
"""
def create_topology(CONFIG_TOPOLOGY):
    ind = Experiment()
    ind, _ = brand_new_experiment(ind, CONFIG_TOPOLOGY.library)
    for i in range(5):
        ind, _ = change_experiment_wrapper(ind, CONFIG_TOPOLOGY.library)
    
    return ind

# %%
"""
Cross-over between two topologies
"""
def crossover_topology(ind1, ind2, CONFIG_TOPOLOGY):
    """
    Currently we do not have a cross-over that works (see NEAT for example of difficulty of this
    As our nodes are heterogeneous, this causes difficulties. Currently, we omit any real cross-over
    """
    ind1 = mutation_topology(ind1)
    ind2 = mutation_topology(ind2)
    return ind1, ind2

#%%
"""
Mutatation of a single topology
"""
def mutation_topology(ind, CONFIG_TOPOLOGY):
    fitness = ind.fitness
    for i in range(5):
        ind, _ = change_experiment_wrapper(ind, CONFIG_TOPOLOGY.library)
        
    ind.fitness = fitness
    return ind,


#%%
"""
Selection of elite individuals out of a population of topologies 
"""
def elite_topology(individuals, NUM_ELITE, NUM_OFFSPRING, CONFIG_TOPOLOGY):
    elite = tools.selBest(individuals, NUM_ELITE)
    offspring = tools.selWorst(individuals, NUM_OFFSPRING)
    return elite, offspring


#%%
"""
Selection of topologies
"""
def selection_topology(individuals, k, CONFIG_TOPOLOGY):
    return tools.selBest(individuals, len(individuals))



#%%
"""
    Measures objective values of a given topology (optimizes control parameters of the topology
"""
def objective_topology(ind, env, CONFIG_TOPOLOGY, CONFIG_PARAMETERS):
    
    #this is an ugly, hacky fix. but for now it works.
    # here we are just making an Experiment class from the Individual class, as there is some issues with pickling Individual class.
    exp = remake_experiment(copy.deepcopy(ind))            
    exp.inject_optical_field(env.field)
    
    exp, hof, hof_fine, log = optimize_experiment(exp, env, CONFIG_PARAMETERS, verbose=False) 
    at = hof_fine[0]
    exp.setattributes(at)
    exp.simulate(env)
    
    field = exp.nodes[exp.measurement_nodes[0]]['output']
    fitness = env.fitness(field)
    ind.inner_attributes = hof_fine[0]

    fig, ax = plt.subplots(1,1)
    exp.draw(node_label='both', title=None, ax=ax)
    plt.show()
    plt.pause(0.1)
    return fitness[0],


#%%
"""
Main optimization function for topologies
"""
def geneticalgorithm_topology(env, CONFIG_TOPOLOGY, CONFIG_PARAMETERS):
    try:
        del(creator.Individual)
        del(creator.FitnessMax)
    except: pass

    creator.create("FitnessMax", base.Fitness, weights=CONFIG_TOPOLOGY.WEIGHTS)
    creator.create("Individual", Experiment, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", create_topology, CONFIG_TOPOLOGY=CONFIG_TOPOLOGY)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", crossover_topology, CONFIG_TOPOLOGY=CONFIG_TOPOLOGY)
    toolbox.register("mutate", mutation_topology, CONFIG_TOPOLOGY=CONFIG_TOPOLOGY)
    toolbox.register("select", selection_topology, CONFIG_TOPOLOGY=CONFIG_TOPOLOGY)
    toolbox.register("elite", elite_topology, CONFIG_TOPOLOGY=CONFIG_TOPOLOGY)

    toolbox.register("evaluate", objective_topology, env=env, CONFIG_TOPOLOGY=CONFIG_TOPOLOGY, CONFIG_PARAMETERS=CONFIG_PARAMETERS)
    
    pop = toolbox.population(n = CONFIG_TOPOLOGY.N_POPULATION)
    
    
    if not CONFIG_TOPOLOGY.INIT:
        pass
    else:
        for i, init in enumerate(CONFIG_TOPOLOGY.INIT):
            pop[i].update(init)
    
    hof = tools.HallOfFame(CONFIG_TOPOLOGY.N_HOF, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # setup variables early, in case of an early termination
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # if CONFIG_TOPOLOGY.MULTIPROC:
    #     pool = mp.Pool(CONFIG_TOPOLOGY.NCORES)
    # else:
    #     pool = None
    pool = None

    # catch exceptions to terminate the optimization early
    # population, logbook = eaSimple(CONFIG_TOPOLOGY, pop, toolbox, pool, logbook,
    #                                cxpb=CONFIG_TOPOLOGY.CRX_PRB,
    #                                mutpb=CONFIG_TOPOLOGY.MUT_PRB,
    #                                ngen=CONFIG_TOPOLOGY.N_GEN,
    #                                stats=stats, halloffame=hof,
    #                                verbose=CONFIG_TOPOLOGY.VERBOSE)
    population, logbook = algorithms.eaSimple(pop, toolbox,
                                              cxpb=CONFIG_TOPOLOGY.CRX_PRB,
                                              mutpb=CONFIG_TOPOLOGY.MUT_PRB,
                                              ngen=CONFIG_TOPOLOGY.N_GEN,
                                              stats=stats, halloffame=hof,
                                              verbose=CONFIG_TOPOLOGY.VERBOSE)

    return hof, population, logbook

