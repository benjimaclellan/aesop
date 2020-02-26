
#%%
import numpy as np
from deap import tools, base, creator
import matplotlib.pyplot as plt
import multiprocess as mp
import copy

#%%
from optimization.geneticalgorithm import eaSimple, initialgeneration, newchildren
from assets.graph_manipulation import change_experiment_wrapper, brand_new_experiment, remake_experiment
from classes.experiment import Experiment
from asope_inner import optimize_experiment
from assets.functions import splitindices


#%% 
"""
Function for creating a New Individual (NA) in the Inner GA
"""
def CREATE_Outer(gapO):
    ind = Experiment()
    ind, _ = brand_new_experiment(ind, gapO.library)
    for i in range(5):
        ind, _ = change_experiment_wrapper(ind, gapO.library)
    
    return ind

#%%
"""
Crosses two individuals in Inner GA
"""
def CX_Outer(ind1, ind2, gapO):
    ind1 = MUT_Outer(ind1)
    ind2 = MUT_Outer(ind2)
    return ind1, ind2

#%%
"""
Mutates a single individual in Inner GA
"""

def MUT_Outer(ind, gapO):  
    fitness = ind.fitness
    for i in range(5):
        ind, _ = change_experiment_wrapper(ind, gapO.library)
        
    ind.fitness = fitness
    return ind,


#%%
"""
Selection criteria for population in Inner GA
"""
def ELITE_Outer(individuals, NUM_ELITE, NUM_OFFSPRING, gapO):
    elite = tools.selBest(individuals, NUM_ELITE)
    offspring = tools.selWorst(individuals, NUM_OFFSPRING)
    return elite, offspring


#%%
"""
Selection criteria for population in Inner GA
"""
def SEL_Outer(individuals, k, gapO):
    return tools.selBest(individuals, len(individuals))



#%%
"""
Fitness function for Inner GA
"""
def FIT_Outer(ind, env, gapO, gapI):
    
    #this is an ugly, hacky fix. but for now it works.
    # here we are just making an Experiment class from the Individual class, as there is some issues with pickling Individual class.
    exp = remake_experiment(copy.deepcopy(ind))            
    exp.inject_optical_field(env.field)
    
    exp, hof, hof_fine, log = optimize_experiment(exp, env, gapI, verbose=False) 
    at = hof_fine[0]
    exp.setattributes(at)
    exp.simulate(env)
    
    field = exp.nodes[exp.measurement_nodes[0]]['output']
    fitness = env.fitness(field)
    ind.inner_attributes = hof_fine[0]
    
    return (fitness[0], -exp.number_of_nodes())


#%%

def outer_geneticalgorithm(env, gapO, gapI):
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
    toolbox.register("attribute", CREATE_Outer, gapO=gapO)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", CX_Outer, gapO=gapO)
    toolbox.register("mutate", MUT_Outer, gapO=gapO)
    toolbox.register("select", SEL_Outer, gapO=gapO)  
    toolbox.register("elite", ELITE_Outer, gapO=gapO)  

    toolbox.register("evaluate", FIT_Outer, env=env, gapO=gapO, gapI=gapI)
    
    pop = toolbox.population(n = gapO.N_POPULATION)
    
    
    if not gapO.INIT:
        pass
    else:
        for i, init in enumerate(gapO.INIT):
            pop[i].update(init)
    
    hof = tools.HallOfFame(gapO.N_HOF, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    stats.register("avg", np.mean)
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

