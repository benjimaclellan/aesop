import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import time
import matplotlib.pyplot as plt
import multiprocess as mp

from assets.functions import extractlogbook, plot_individual
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator
from assets.classes import Experiment, GeneticAlgorithmParameters

from optimization.ga_functions_inner import inner_geneticalgorithm
from optimization.graddescent import finetune_individual

plt.close("all")

## ************************************************

if __name__ == '__main__':  
    env = PulseEnvironment()

    components = [AWG(), Fiber()]
    experiment = Experiment()
    experiment.buildexperiment(components)
        
    gap = GeneticAlgorithmParameters()
    gap.NFITNESS = 2
    gap.WEIGHTS = (1.0, 0.5)
    gap.MULTIPROC = True
    gap.NCORES = mp.cpu_count()
    gap.N_POPULATION = 100      # number of individuals in a population
    gap.N_GEN = 50              # number of generations
    gap.MUT_PRB = 0.7           # independent probability of mutation
    gap.CRX_PRB = 0.7           # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 1             # verbose print statement for GA statistics
    
    tstart = time.time()
    hof, population, logbook = inner_geneticalgorithm(gap, env, experiment)
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
        
        env.reset()
        experiment.setattributes(individual)
        experiment.simulate(env)
        
        fitness = env.fitness()
        
        fig, ax = plot_individual(env, fitness)
        
        print(individual)
        
        ## Now fine tune the best of the hall of fame
#        env.reset()
#        individual = finetune_individual(individual, gap, env, experiment, sim)
#        sim.simulate_experiment(individual, experiment, env, verbose=True)
#        fitness = sim.fitness(env)
#
#        fig, ax = plot_individual(individual, fitness, env, sim)
#        plt.show()
