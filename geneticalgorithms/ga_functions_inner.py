import numpy as np
from deap import tools, base, creator
import random
import multiprocess as mp

## --------------------------------------------------------------------
"""
Function for creating a New Individual (NA) in the Inner GA
"""
def CREATE_Inner(experiment):
    ind = []
    for component in experiment:        
        ind.append(NA_Inner(component))
    return ind



## --------------------------------------------------------------------
"""
Function for creating a New Attribute (NA) in the Inner GA
"""
def NA_Inner(c):

    if c.type == 'fiber':
        at = [randomattribute(c.LOWER[0], c.UPPER[0], c.DTYPE[0], c.DSCRTVAL[0])]

    elif c.type == 'awg':
        n = randomattribute(c.LOWER[0], c.UPPER[0], c.DTYPE[0], c.DSCRTVAL[0])
        vals = []
        for i in range(n):
            vals.append(randomattribute(c.LOWER[1], c.UPPER[1], c.DTYPE[1], c.DSCRTVAL[1]))
        at = [n] + vals

    elif c.type == 'phasemodulator':
        at = []
        for i in range(len(c.LOWER)):
            at.append(randomattribute(c.LOWER[i], c.UPPER[i], c.DTYPE[i], c.DSCRTVAL[i]))
    else: 
        raise ValueError('No such component type')
        
    return at

def randomattribute(low=0.0, high=1.0, dtypes='float', dscrtval=None):
    if dtypes == 'float':
        if dscrtval is not None:  
            at = round(np.random.uniform( low, high )/dscrtval) * dscrtval 
        else:
            at = np.random.uniform( low, high )
            
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
#    print(ind1)
#    print(ind2)
    
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
            
#    print(ind1)
#    print(ind2)
    
    return ind1, ind2





## --------------------------------------------------------------------
"""
Mutates a single individual in Inner GA
"""

def MUT_Inner(experiment, ind):
#    print('ind = {}'.format(ind))
    
    mut_comp = np.random.randint(0,len(experiment))    
    c = experiment[mut_comp]
    
    if c.type == 'fiber':
        ind[mut_comp][0] = randomattribute(c.LOWER[0], c.UPPER[0], c.DTYPE[0], c.DSCRTVAL[0])

    elif c.type == 'awg':
        mut_loc = np.random.randint(0, len(ind[mut_comp]))
        if mut_loc == 0: # mutates the number of steps
            n_mut = randomattribute(c.LOWER[0], c.UPPER[0], c.DTYPE[0], c.DSCRTVAL[0])
            if n_mut > ind[mut_comp][0]: # mutated to have more steps than before
                new_vals = []
                for i in range(n_mut-ind[mut_comp][0]):
                    new_vals.append(randomattribute(c.LOWER[1], c.UPPER[1], c.DTYPE[1], c.DSCRTVAL[1]))
                vals = ind[mut_comp][1:] + new_vals
                ind[mut_comp] = [n_mut] + vals
            else: 
                vals = ind[mut_comp][1:n_mut+1]
                ind[mut_comp] = [n_mut] + vals
        else:
            ind[mut_comp][mut_loc] = randomattribute(c.LOWER[1], c.UPPER[1], c.DTYPE[1], c.DSCRTVAL[1])

    elif c.type == 'phasemodulator':
        mut_loc = np.random.randint(0, len(ind[mut_comp]))
        ind[mut_comp][mut_loc] = randomattribute(c.LOWER[mut_loc], c.UPPER[mut_loc], c.DTYPE[mut_loc], c.DSCRTVAL[mut_loc])
        
    else: 
        raise ValueError('No such component type')
    
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
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", CREATE_Inner, experiment)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", CX_Inner)
    toolbox.register("mutate", MUT_Inner, experiment)
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

    assert cxpb < 1.0 and mutpb < 1.0
    
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
                args = [offspring[splt_indx[i]:splt_indx[i+1]], toolbox, cxpb, mutpb]
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
def split_indices(num, div):
    indices = [0]
    for i in range(div):
        val = num//(div - i)
        num += -val
        prev = indices[-1]
        indices.append(val + prev)
    return indices
