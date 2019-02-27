import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import time
import matplotlib.pyplot as plt
import multiprocess as mp
import uuid

from assets.functions import extractlogbook, plot_individual, save_experiment, load_experiment
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper
from assets.classes import Experiment, GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

## ************************************************

def main(filename = None):                  
    if filename == None: save = False
#    filename = 'results/' + str(uuid.uuid4().hex)
#    filename = 'results/' + time.strftime("%Y_%m_%d-%H_%M_%S")
    
    env = PulseEnvironment(p = 2, q = 1)
    
    components = [AWG(), Fiber()]
    experiment = Experiment()
    experiment.buildexperiment(components)
        
    gap = GeneticAlgorithmParameters()
    gap.NFITNESS = 2
    gap.WEIGHTS = (1.0,0.1)
    gap.MULTIPROC = True
    gap.NCORES = mp.cpu_count()
    gap.N_POPULATION = 200      # number of individuals in a population
    gap.N_GEN = 50              # number of generations
    gap.MUT_PRB = 0.2           # independent probability of mutation
    gap.CRX_PRB = 0.95          # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 1             # verbose print statement for GA statistics
    
    print('Number of cores: {}, number of generations: {}, size of population: {}'.format(gap.NCORES, gap.N_GEN, gap.N_POPULATION))
    
    tstart = time.time()
    hof, population, logbook = inner_geneticalgorithm(gap, env, experiment)
    tstop = time.time()

    log = extractlogbook(logbook)    
#    plt.figure()
#    plt.plot(log['gen'], log['max'], label='max') 
#    plt.plot(log['gen'], log['min'], label='min') 
#    plt.plot(log['gen'], log['avg'], label='avg') 
#    plt.legend()
    
    print('\nElapsed time = {}'.format(tstop-tstart))
    print('Total number of individuals measured: {}\n'.format(sum(log['nevals'])))
    
    """
    Now we visualizing the best HOF individual found, and slightly improve it
    """    
    for j in range(gap.N_HOF):
        individual = hof[j]
        print(individual)
        
        env.reset()
        experiment.setattributes(individual)
        experiment.simulate(env)
        fitness = env.fitness()
        plot_individual(env, fitness)
        
        # Now fine tune the best of the hall of fame
        individual = finetune_individual(individual, env, experiment)
        env.reset()
        print(individual)
        experiment.setattributes(individual)
        experiment.simulate(env)
        
        fitness = env.fitness()
        plot_individual(env, fitness)
        plt.show()
        
        experiment.visualize(env)
        
        if save:
            save_experiment(filename, experiment, env)
        
        return experiment, env

if __name__ == '__main__': 
    (experiment, env) = main()
