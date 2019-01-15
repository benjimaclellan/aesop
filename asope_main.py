import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import time
import matplotlib.pyplot as plt
import scipy.optimize as opt
import multiprocess as mp

#from components.component_parameters import component_parameters

from classes import GeneticAlgorithmParameters
from functions.helperfunctions import extract_bounds, buildexperiment, extractlogbook
#from functions.helperfunctions import experiment_description

from environments.environment_pulse import PulseEnvironment
from simulators.simulator_classical import ClassicalSimulator

from geneticalgorithms.ga_functions_inner import inner_geneticalgorithm, FIT_Inner

plt.close("all")

## ************************************************
def final_opt(optlist, ind, idx, gap, env, experiment, sim):    
    ind = unpack_opt(ind, optlist, idx)
    fitness = -FIT_Inner(ind, gap, env, experiment, sim)[1]
    return fitness

def pack_opt(ind, experiment):
    idx = []
    optlist = []
    for i in range(len(experiment)): 
        c = experiment[i]
        for j in range(c.FINETUNE_SKIP, len(ind[i])):
            idx.append( [ i, j ] )
            print(ind[i][j])
            optlist.append(ind[i][j])
    return optlist, idx

def unpack_opt(ind, optlist, idx):
    for i in range(len(idx)):
        ind[idx[i][0]][idx[i][1]] = optlist[i]
    return ind
        
def finetune_individual(ind, gap, env, experiment, sim):
    (optlist, idx) = pack_opt(ind, experiment)
    
    optres = opt.minimize(final_opt, optlist, args=(ind, idx, gap, env, experiment, sim), method='CG', options={'maxiter':1000})    
    
    ind = unpack_opt(ind, optres.x.tolist(), idx)
    return ind

if __name__ == '__main__':  
    
    """   
    ASOPE V1 is a program which creates an optimal experimental setup for a desired outcome. The software will generate a random setup from the available component, and optimize parameters
    TODO:
        Implement these, such that the next run includes one individual that uses the best values from the previous run
            USE_PREVIOUS_HOF  = False
            SAVE_PREVIOUS_HOF = False
    """     
    

    fitnessfunction = 'TemporalTalbot'
    sim_kwargs = {'p':2, 'q':1}

    env = PulseEnvironment()
    sim = ClassicalSimulator(fitnessfunction, **sim_kwargs)

    experiment_ids = [1,0]
    experiment = buildexperiment(experiment_ids)

    (N_ATTRIBUTES, BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS) = extract_bounds(experiment)
    
    gap = GeneticAlgorithmParameters(N_ATTRIBUTES, BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS)
    
    gap.NFITNESS = 2
    gap.WEIGHTS = (1.0, 0.001)
    gap.MULTIPROC = True
    gap.NCORES = mp.cpu_count()
    gap.N_POPULATION = 300      # number of individuals in a population
    gap.N_GEN = 50             # number of generations
    gap.MUT_PRB = 0.7           # independent probability of mutation
    gap.CRX_PRB = 0.7           # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 1             # verbose print statement for GA statistics
    
    tstart = time.time()
    hof, population, logbook = inner_geneticalgorithm(gap, env, experiment, sim)
    tstop = time.time()

    log = extractlogbook(logbook)    
    
    plt.figure()
    plt.plot(log['gen'], log['max'])
    
    print('\nElapsed time = {}'.format(tstop-tstart))
    print('Total number of individuals measured: {}\n'.format(sum(log['nevals'])))
    
    """
    Now we visualizing the best HOF individual found, and slightly improve it
    """    
    for j in range(gap.N_HOF):
        individual = hof[j]
        env.reset()
        sim.simulate_experiment(individual, experiment, env, verbose=False)
        fitness = sim.fitness(env)
        print(fitness)

        fig, ax = plt.subplots(2,1)
        ax[0].plot(env.t, env.P(env.At0), label='initial')
        ax[1].plot(env.f, env.PSD(env.Af0, env.df))

        ax[0].plot(env.t, env.P(env.At), label='final')
        ax[1].plot(env.f, env.PSD(env.Af, env.df))

        ax[0].set_title('Fitness {}'.format(fitness))
        ax[0].legend()
        plt.show()
        
        
        
        individual = hof[j] 
        individual = finetune_individual(individual, gap, env, experiment, sim)
#        
        env.reset()
        sim.simulate_experiment(individual, experiment, env, verbose=True)
        fitness = sim.fitness(env)

        fig, ax = plt.subplots(2,1)
        ax[0].plot(env.t, env.P(env.At0), label='initial')
        ax[1].plot(env.f, env.PSD(env.Af0, env.df))

        ax[0].plot(env.t, env.P(env.At), label='final')
        ax[1].plot(env.f, env.PSD(env.Af, env.df))

        ax[0].set_title('Fitness {}'.format(fitness))
        ax[0].legend()
        plt.show()
