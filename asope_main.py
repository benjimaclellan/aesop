import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import time
import matplotlib.pyplot as plt
from visualization.visualize import plot_individual
import multiprocess as mp

from classes import GeneticAlgorithmParameters
from functions.helperfunctions import extract_bounds, buildexperiment, extractlogbook

from environments.environment_pulse import PulseEnvironment
from simulators.simulator_classical import ClassicalSimulator

from geneticalgorithms.ga_functions_inner import inner_geneticalgorithm
from graddescent.graddescent import finetune_individual
plt.close("all")

## ************************************************

if __name__ == '__main__':  
    fitnessfunction = 'TalbotEffect'
    sim_kwargs = {'p':4, 'q':1}

    env = PulseEnvironment()
    sim = ClassicalSimulator(fitnessfunction, **sim_kwargs)

    experiment_ids = [1,0]
    experiment = buildexperiment(experiment_ids)

    (N_ATTRIBUTES, BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS) = extract_bounds(experiment)
    
    gap = GeneticAlgorithmParameters(N_ATTRIBUTES, BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS)
    
    gap.NFITNESS = 2
    gap.WEIGHTS = (1.0, 0.5)
    gap.MULTIPROC = True
    gap.NCORES = mp.cpu_count()
    gap.N_POPULATION = 1000      # number of individuals in a population
    gap.N_GEN = 100              # number of generations
    gap.MUT_PRB = 0.7           # independent probability of mutation
    gap.CRX_PRB = 0.7           # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 1             # verbose print statement for GA statistics
    
    tstart = time.time()
    hof, population, logbook = inner_geneticalgorithm(gap, env, experiment, sim)
    tstop = time.time()

    log = extractlogbook(logbook)    
#    plt.figure()
#    plt.plot(log['gen'], log['max']) 
    
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

        fig, ax = plot_individual(individual, fitness, env, sim)
        
        ## Now fine tune the best of the hall of fame
        env.reset()
        individual = finetune_individual(individual, gap, env, experiment, sim)
        sim.simulate_experiment(individual, experiment, env, verbose=True)
        fitness = sim.fitness(env)

        fig, ax = plot_individual(individual, fitness, env, sim)
        plt.show()
