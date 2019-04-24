from deap import tools, base, creator
import random
from assets.functions import splitindices

#%%
def eaSimple(gap, population, toolbox, pool, logbook, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    assert cxpb <= 1.0 and mutpb <= 1.0
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    
    if gap.MULTIPROC:
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
        
        NUM_NEW = len(population) - gap.NUM_ELITE - gap.NUM_MATE_POOL
        
        tmp = tools.selBest(population, gap.NUM_ELITE + gap.NUM_MATE_POOL)
        elite = tools.selBest(tmp, gap.NUM_ELITE)
        
        offspring = tools.selWorst(tmp, gap.NUM_MATE_POOL)
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

#%%
def initialgeneration(args):
    (invalid_ind, toolbox) = args
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit        
    
    return invalid_ind

#%%
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
    

#%%
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

