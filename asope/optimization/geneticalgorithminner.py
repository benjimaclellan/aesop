import numpy as np

from deap import tools, base, creator, algorithms
import random
# import multiprocess as mp
import multiprocessing
import copy
from optimization.geneticalgorithm import eaSimple

# Workaround for multiprocessing with Windows (solution from https://github.com/DEAP/deap/issues/268)
from loky import get_reusable_executor
def prepare_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
    # creator.create("FitnessMax", base.Fitness, weights=gapI.WEIGHTS)
    creator.create("Individual", dict, fitness=creator.FitnessMax)
prepare_creator()

#%%
"""
Function for creating a New Individual (NA) in the Inner GA
"""
def CREATE_Inner(experiment):
    ind = experiment.newattributes()
    return ind

#%%
"""
Crosses two individuals in Inner GA
"""
def CX_Inner(ind1, ind2):
    
    keys = list(ind1.keys())
        
    if len(keys) == 0:
        raise ValueError
    elif len(keys) == 1:
        ind1, ind2 = copy.deepcopy(ind2), copy.deepcopy(ind1)
        
    elif len(keys) >= 2:
        cx_split = random.randint(1,len(keys)-1)
    
        keysa = keys[0:cx_split]
        keysb = keys[cx_split:]
        
        for key in keysa: ind1[key] = ind2[key]
        for key in keysb: ind2[key] = ind1[key]
        

    return ind1, ind2

#%%
"""
Mutates a single individual in Inner GA
"""

def MUT_Inner(experiment, ind):  
    mut_node = random.choice(list(ind))  
    ind[mut_node] = experiment.nodes[mut_node]['info'].mutate()
    return ind,


#%%
"""
Selection criteria for population in Inner GA
"""
def ELITE_Inner(individuals, NUM_ELITE, NUM_OFFSPRING):
    elite = tools.selBest(individuals, NUM_ELITE)
    offspring = tools.selWorst(individuals, NUM_OFFSPRING)
    return elite, offspring


#%%
"""
Selection criteria for population in Inner GA
"""
def SEL_Inner(individuals, k):
    return tools.selNSGA2(individuals, k)
    # return tools.selBest(individuals, len(individuals))



#%%
"""
Fitness function for Inner GA
"""
def FIT_Inner(ind, env, experiment, UDR = False):
        
    experiment.setattributes(ind)
    experiment.simulate(env)

    measurement_node = experiment.measurement_nodes[0]
    field = experiment.nodes[measurement_node]['output']  #.reshape(env.N)
    opt_fit = env.fitness(field)[0]
    return [opt_fit]

#%%
def inner_geneticalgorithm(gapI, env, experiment):
    """
    Here, we set up our inner genetic algorithm. This will eventually be moved to a different function/file to reduce clutter
    """


    # try:
    #     del(creator.Individual)
    #     del(creator.FitnessMax)
    # except AttributeError:
    #     pass

    toolbox = base.Toolbox()

    if gapI.MULTIPROC:
        # The following code was altered to use LOKY's reusable processes, using workaround mentioned above
        pool = get_reusable_executor(gapI.NCORES, reuse=True)
        # After initiating the pool, we define a "supermap" function that guarantees that every process will execute the creator preparation procedure and then run the mapped function
        def supermap(*args, **kwargs):
            prepare_creator()
            return pool.map(*args, **kwargs)
        # We then register our "supermap" function as the parallel directive for DEAP
        toolbox.register("map", supermap)
    else:
        pool = None
        prepare_creator()

    # creator_test.create("FitnessMax", base.Fitness, weights=gapI.WEIGHTS)
    # creator_test.create("Individual", dict, fitness=creator_test.FitnessMax)

    toolbox.register("attribute", CREATE_Inner, experiment)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", CX_Inner)
    toolbox.register("mutate", MUT_Inner, experiment)
    toolbox.register("select", SEL_Inner)
    toolbox.register("elite", ELITE_Inner)
    toolbox.register("evaluate", FIT_Inner, env=env, experiment=experiment)

    pop = toolbox.population(n = gapI.N_POPULATION)

    if not gapI.INIT:
        pass
    else:
        for i, init in enumerate(gapI.INIT):
            pop[i].update(init)

    hof = tools.HallOfFame(gapI.N_HOF, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("Average [fitness, variance]", np.mean, axis = 0)
    stats.register("Best [fitness, variance]", np.max, axis = 0)

    # setup variables early, in case of an early termination
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    population, logbook = algorithms.eaSimple(pop, toolbox, cxpb=gapI.CRX_PRB, mutpb=gapI.MUT_PRB, ngen=gapI.N_GEN, stats=stats, halloffame=hof, verbose=gapI.VERBOSE)
#     population, logbook = eaSimple(gapI, pop, toolbox, pool, logbook, cxpb=gapI.CRX_PRB, mutpb=gapI.MUT_PRB, ngen=gapI.N_GEN, stats=stats, halloffame=hof, verbose=gapI.VERBOSE)

    return hof, population, logbook

