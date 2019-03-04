import numpy as np
from deap import tools, base, creator
import random
import multiprocess as mp
from assets.functions import splitindices

## --------------------------------------------------------------------
"""
Function for creating a New Individual (NA) in the Inner GA
"""
def CREATE_Inner(experiment):
    ind = experiment.newattributes()
    return ind

## --------------------------------------------------------------------
"""
Crosses two individuals in Inner GA
"""
def CX_Inner(ind1, ind2):
    
#    a = ind1
#    b = ind2 
#    ind1 = b
#    ind2 = a
    keys = list(ind1.keys())
    
    if len(keys) == 0:
        raise ValueError
    elif len(keys) == 1:
        ind1, ind2 = ind2.copy(), ind1.copy()
    elif len(keys) >= 2:
        cx_split = random.randint(1,len(keys)-1)
    
        keysa = keys[0:cx_split]
        keysb = keys[cx_split:]
        
        for key in keysa: ind1[key] = ind2[key]
        for key in keysb: ind2[key] = ind1[key]
        
#    size = len(ind1)
#    if size == 1:
#        ind1[:], ind2[:] = ind2[:].copy(), ind1[:].copy()   
#    else:
#        cxpoint1 = random.randint(1, size)
#        cxpoint2 = random.randint(1, size - 1)
#        if cxpoint2 >= cxpoint1:
#            cxpoint2 += 1
#        else: # Swap the two cx points
#            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
#    
#        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
#            = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()   
    
    return ind1, ind2

## --------------------------------------------------------------------
"""
Mutates a single individual in Inner GA
"""

def MUT_Inner(experiment, ind):  
    mut_node = random.choice(list(ind))  
    ind[mut_node] = experiment.nodes[mut_node]['info'].mutate()
    return ind,


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
def FIT_Inner(individual, env, experiment):

    experiment.setattributes(individual)
    experiment.simulate(env)
    
    measurement_node = experiment.measurement_nodes[0]
    At = experiment.nodes[measurement_node]['output'].reshape(env.N)
    fitness = env.fitness(At)
    
    return fitness[0]*fitness[1],


## --------------------------------------------------------------------
## --------------------------------------------------------------------
## --------------------------------------------------------------------
def inner_geneticalgorithm(gap, env, experiment):
    """
    Here, we set up our inner genetic algorithm. This will eventually be moved to a different function/file to reduce clutter
    """    
    try: 
        del(creator.Individual) 
        del(creator.FitnessMax)
    except: pass

    creator.create("FitnessMax", base.Fitness, weights=gap.WEIGHTS)
    creator.create("Individual", dict, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", CREATE_Inner, experiment)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", CX_Inner)
    toolbox.register("mutate", MUT_Inner, experiment)
    toolbox.register("select", SEL_Inner)  
    toolbox.register("elite", ELITE_Inner)  

    toolbox.register("evaluate", FIT_Inner, env=env, experiment=experiment)
    
    pop = toolbox.population(n = gap.N_POPULATION)
    
    hof = tools.HallOfFame(gap.N_HOF, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    stats.register("avg", np.mean)
#    stats.register("std", np.std)
#    stats.register("min", np.min)
    stats.register("max", np.max)
    
    population, logbook = eaSimple(gap, pop, toolbox, cxpb=gap.CRX_PRB, mutpb=gap.MUT_PRB, ngen=gap.N_GEN, stats=stats, halloffame=hof, verbose=gap.VERBOSE)

    return hof, population, logbook


def eaSimple(gap, population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    assert cxpb < 1.0 and mutpb < 1.0
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    
    if gap.MULTIPROC:
        pool = mp.Pool(gap.NCORES)

        splt_indx = splitindices(len(invalid_ind), gap.NCORES) 
                       
        mp_input = []
        for i in range(0,gap.NCORES):
            args = [invalid_ind[splt_indx[i]:splt_indx[i+1]], toolbox]
            mp_input.append(args)
        
        results = pool.map(initialgeneration, mp_input)
        
        population = []
        for i in range(0,gap.NCORES):
            population += results[i]
    else:
        args = [invalid_ind, toolbox]
        population = initialgeneration(args)


    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        
        
        NUM_ELITE = 2
        NUM_MATE_POOL = 3 * len(population) // 4
        
        NUM_NEW = len(population) - NUM_MATE_POOL - NUM_ELITE
        
#        NUM_OFFSPRING = len(population) - NUM_MATE_POOL - NUM_ELITE
        tmp = tools.selBest(population, NUM_ELITE + NUM_MATE_POOL)
        elite = tools.selBest(tmp, NUM_ELITE)
        
        offspring = tools.selWorst(tmp, NUM_MATE_POOL)
        random.shuffle(offspring)
        
        new = tools.selWorst(population, NUM_NEW)
#        offspring = tools.selWorst(population, NUM_OFFSPRING)
        
        if gap.MULTIPROC:
            splt_indx = splitindices(len(offspring), gap.NCORES)                        
            mp_input = []

            for i in range(0,gap.NCORES):
                args = [offspring[splt_indx[i]:splt_indx[i+1]], toolbox, cxpb, mutpb]
                mp_input.append(args)
                                    
            results = pool.map(varychildren, mp_input)
            
            offspring = []
            for i in range(0,gap.NCORES):
                offspring += results[i][:]
                
            ### now create new individuals
            splt_indx = splitindices(len(new), gap.NCORES)                        
            mp_input = []

            for i in range(0,gap.NCORES):
                args = [new[splt_indx[i]:splt_indx[i+1]], toolbox]
                mp_input.append(args)
                                    
            results = pool.map(newchildren, mp_input)
            
            new = []
            for i in range(0,gap.NCORES):
                new += results[i][:]    
            
                
        else:
            args = (offspring, toolbox, cxpb, mutpb)
            offspring = varychildren(args)
            
            args = (offspring, toolbox)
            offspring = newchildren(args)
            
        population[:] = elite + offspring + new
    
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

def initialgeneration(args):
    (invalid_ind, toolbox) = args
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    return invalid_ind

def newchildren(args):
    (offspring, toolbox) = args
    for i in range(len(offspring)):
        offspring[i] = toolbox.individual()
        del offspring[i].fitness.values    
        
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return offspring
    


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

