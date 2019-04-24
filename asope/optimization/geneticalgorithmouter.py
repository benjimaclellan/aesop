import numpy as np
from deap import tools, base, creator
#import random
import matplotlib.pyplot as plt
import multiprocess as mp
import copy
from optimization.geneticalgorithm import eaSimple, initialgeneration, newchildren
from assets.graph_manipulation import change_experiment_wrapper, brand_new_experiment, remake_experiment
from assets.classes import Experiment
from asope_main import optimize_experiment
from assets.functions import splitindices
import random


#%% 
"""
Function for creating a New Individual (NA) in the Inner GA
"""
def CREATE_Outer():
    ind = Experiment()
    ind, _ = brand_new_experiment(ind)
    
    
    for i in range(5):
        ind, _ = change_experiment_wrapper(ind)
    
    return ind

#%%
"""
Crosses two individuals in Inner GA
"""
def CX_Outer(ind1, ind2):
    print('Am I even crossing the fourth wall tho?')
    ind1 = MUT_Outer(ind1)
    ind2 = MUT_Outer(ind2)
    return ind1, ind2

#%%
"""
Mutates a single individual in Inner GA
"""

def MUT_Outer(ind):  
    fitness = ind.fitness
    
    for i in range(20):
        ind, _ = change_experiment_wrapper(ind)
        
    ind.fitness = fitness
    return ind,


#%%
"""
Selection criteria for population in Inner GA
"""
def ELITE_Outer(individuals, NUM_ELITE, NUM_OFFSPRING):
    elite = tools.selBest(individuals, NUM_ELITE)
    offspring = tools.selWorst(individuals, NUM_OFFSPRING)
    return elite, offspring


#%%
"""
Selection criteria for population in Inner GA
"""
def SEL_Outer(individuals, k):
    return tools.selBest(individuals, len(individuals))



#%%
"""
Fitness function for Inner GA
"""
def FIT_Outer(ind, env):
    
    #this is an ugly, hacky fix. but for now it works.
    # here we are just making an Experiment class from the Individual class, as there is some issues with pickling Individual class.
    experiment = remake_experiment(copy.deepcopy(ind))
    
    # here we initialize what pulse we inject into each starting node
    for node in experiment.nodes():
        if len(experiment.pre(node)) == 0:
            experiment.nodes[node]['input'] = env.At
            
    experiment, hof, hof_fine, hof_fine_fitness = optimize_experiment(experiment, env)

    fitness = hof_fine_fitness[0]
    
    ind.inner_attributes = hof_fine[0]
    
#    plt.figure()
#    experiment.draw(node_label='both')
    
    return fitness[0], fitness[1],


#%%

def outer_geneticalgorithm(gapO, env):
    """
    Here, we set up our inner genetic algorithm. This will eventually be moved to a different function/file to reduce clutter
    """    
    try: 
        del(creator.Individual) 
        del(creator.FitnessMax)
    except: pass

    creator.create("FitnessMax", base.Fitness, weights=gapO.WEIGHTS)
    creator.create("Individual", Experiment, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", CREATE_Outer)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
#    toolbox.register("individual", creator.Individual)
#    toolbox.register("individual", CREATE_Outer)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", CX_Outer)
    toolbox.register("mutate", MUT_Outer)
    toolbox.register("select", SEL_Outer)  
    toolbox.register("elite", ELITE_Outer)  

    toolbox.register("evaluate", FIT_Outer, env=env)
    
    pop = toolbox.population(n = gapO.N_POPULATION)
    
    
    if not gapO.INIT:
        pass
    else:
        for i, init in enumerate(gapO.INIT):
            pop[i].update(init)
    
    hof = tools.HallOfFame(gapO.N_HOF, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
#    stats.register("avg", np.mean)
#    stats.register("std", np.std)
#    stats.register("min", np.min)
    stats.register("max", np.max)

    # setup variables early, in case of an early termination
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    if gapO.MULTIPROC:
        pool = mp.Pool(gapO.NCORES)
    else: 
        pool = None
    
    # catch exceptions to terminate the optimization early
    population, logbook = eaSimple(gapO, pop, toolbox, pool, logbook, cxpb=gapO.CRX_PRB, mutpb=gapO.MUT_PRB, ngen=gapO.N_GEN, stats=stats, halloffame=hof, verbose=gapO.VERBOSE)

    return hof, population, logbook

