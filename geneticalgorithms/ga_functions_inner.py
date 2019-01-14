import numpy as np
from numpy import pi
from deap import algorithms, tools, base, creator
import random
import scipy.optimize as opt
import time

from copy import copy, deepcopy

#import settings

import multiprocess as mp


## --------------------------------------------------------------------
"""
Function for creating a New Individual (NA) in the Inner GA
"""
def CREATE_Inner(low=0.0, high=1.0, dtypes='float', dscrtval=None):
    assert len(low) == len(high) == len(dtypes) == len(dscrtval)
    ind = []
    for i_att in range(len(low)):
        ind.append( NA_Inner(low=low[i_att], high=high[i_att], dtypes=dtypes[i_att], dscrtval=dscrtval[i_att]) )
    return ind



## --------------------------------------------------------------------
"""
Function for creating a New Attribute (NA) in the Inner GA
"""
def NA_Inner(low=0.0, high=1.0, dtypes='float', dscrtval=None):
#    assert len(low) == len(high) == len(dtypes) == len(dscrtval)
    
    if dtypes == 'float':
        if dscrtval is not None:  
            at = np.round(np.random.uniform( low, high )/dscrtval) * dscrtval 
        else:
            at = np.random.uniform( low, high)
            
    elif dtypes == 'int':
        if dscrtval is not None:    
            at = np.round(np.random.randint( low/dscrtval, high/dscrtval))*dscrtval
        else: 
            at = np.random.randint(low, high)
    else:
        raise ValueError('Unknown datatype when making a new attribute')

    return at




## --------------------------------------------------------------------
"""
Crosses two individuals in Inner GA
"""
def CX_Inner(ind1, ind2):
    size = len(ind1)
    if size == 1:
        ind1[:], ind2[:] = ind2[:].copy(), ind1[:].copy()   
    else:
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else: # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()    
    return ind1, ind2





## --------------------------------------------------------------------
"""
Mutates a single individual in Inner GA
"""
def MUT_Inner(individual, low=0.0, high=1.0, dtypes='float', dscrtval=None):
    for i_att in range(len(individual)):
        if np.random.rand() > 0:
            individual[i_att] = NA_Inner(low=low[i_att], high=high[i_att], dtypes=dtypes[i_att], dscrtval=dscrtval[i_att])
    return individual,





## --------------------------------------------------------------------
"""
Selection criteria for population in Inner GA
"""
def ELITE_Inner(individuals, NUM_ELITE, NUM_OFFSPRING):
    elite = tools.selBest(individuals, NUM_ELITE)
    offspring = tools.selWorst(individuals, NUM_OFFSPRING)
    return elite, offspring


## --------------------------------------------------------------------
"""
Selection criteria for population in Inner GA
"""
def SEL_Inner(individuals, k):
    return tools.selBest(individuals, len(individuals))



## --------------------------------------------------------------------
"""
Fitness function for Inner GA
"""
def FIT_Inner(individual, gap=None, env=None, experiment=None, sim=None):
    if env==None or gap==None or experiment==None or sim==None:
        raise ValueError('All classes must be passed to the fitness function')

    sim.simulate_experiment(individual, experiment, env)
    fitness = sim.fitness(env)[0:gap.NFITNESS]

    return tuple(fitness)


## --------------------------------------------------------------------
## --------------------------------------------------------------------
## --------------------------------------------------------------------
def inner_geneticalgorithm(gap, env, experiment, sim):
    """
    Here, we set up our inner genetic algorithm. This will eventually be moved to a different function/file to reduce clutter
    """    
    try: 
        del(creator.Individual) 
        del(creator.FitnessMax)
    except: pass

    creator.create("FitnessMax", base.Fitness, weights=gap.WEIGHTS)
#    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", CREATE_Inner, low=gap.BOUNDSLOWER, high=gap.BOUNDSUPPER, dtypes=gap.DTYPES, dscrtval=gap.DSCRTVALS)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", CX_Inner)
    toolbox.register("mutate", MUT_Inner, low=gap.BOUNDSLOWER, high=gap.BOUNDSUPPER, dtypes=gap.DTYPES, dscrtval=gap.DSCRTVALS)
    toolbox.register("select", SEL_Inner)  
    toolbox.register("elite", ELITE_Inner)  

    toolbox.register("evaluate", FIT_Inner, gap=gap, env=env, experiment=experiment, sim=sim)
    
    pop = toolbox.population(n = gap.N_POPULATION)
    
    hof = tools.HallOfFame(gap.N_HOF, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
#    stats.register("avg", np.mean)
#    stats.register("std", np.std)
#    stats.register("min", np.min)
    stats.register("max", np.max)
    
    population, logbook = eaSimple(gap, pop, toolbox, cxpb=gap.CRX_PRB, mutpb=gap.MUT_PRB, ngen=gap.N_GEN, stats=stats, halloffame=hof, verbose=gap.VERBOSE)


    return hof, population, logbook


def eaSimple(gap, population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    assert (cxpb + mutpb) < 1.0
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    
    if gap.MULTIPROC:
        pool = mp.Pool(gap.NCORES)

        splt_indx = split_indices(len(invalid_ind), gap.NCORES) 
                       
        mp_input = []
        for i in range(0,gap.NCORES):
            args = [invalid_ind[splt_indx[i]:splt_indx[i+1]], toolbox]
            mp_input.append(args)
        
        results = pool.map(mp_gen0, mp_input)
        
        population = []
        for i in range(0,gap.NCORES):
            population += results[i]
    else:
        args = [invalid_ind, toolbox]
        population = mp_gen0(args)


    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        
        NUM_ELITE = 4
        NUM_OFFSPRING = len(population) - NUM_ELITE
        
        elite = tools.selBest(population, NUM_ELITE)
        offspring = tools.selWorst(population, NUM_OFFSPRING)
        
        if gap.MULTIPROC:            
            splt_indx = split_indices(len(offspring), gap.NCORES)                        
            mp_input = []

            for i in range(0,gap.NCORES):
                args = [deepcopy(offspring[splt_indx[i]:splt_indx[i+1]]), deepcopy(toolbox), cxpb, mutpb]
                mp_input.append(args)
                                    
            results = pool.map(varychildren, mp_input)
            
            offspring = []
            for i in range(0,gap.NCORES):
                offspring += results[i][:]
                
        else:
            args = (offspring, toolbox, cxpb, mutpb)
            offspring = varychildren(args)
            
        population[:] = elite + offspring
    
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    
    if gap.MULTIPROC: 
        pool.close()        
    
    return population, logbook

## ------------------------------

def mp_gen0(args):
    (invalid_ind, toolbox) = args
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    return invalid_ind


#def ga_mp():
    


def varychildren(args):
    
    (offspring, toolbox, cxpb, mutpb) = args

    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
    
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    return offspring



##### ---------------------------------
def final_opt(ind, toolbox):
    fitness = -toolbox.evaluate(ind)[1]
    return fitness
    
def split_indices(num, div):
    indices = [0]
    for i in range(div):
        val = num//(div - i)
        num += -val
        prev = indices[-1]
        indices.append(val + prev)
    return indices
