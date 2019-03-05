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
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from assets.classes import Experiment, GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

## ************************************************

if __name__ == '__main__': 
#    """
    filename = None
    if filename == None: save = False
#    filename = 'results/' + str(uuid.uuid4().hex)
#    filename = 'results/' + time.strftime("%Y_%m_%d-%H_%M_%S")
    
    env = PulseEnvironment(p = 2, q = 1)

    components = (
        {
         0:AWG(),
         1:Fiber(),
        })
    adj = [(0,1)]
    
    # note that the fitness function is evaluated at the first measurement_node
    measurement_nodes = [1]
    
    
    experiment = Experiment()
    experiment.buildexperiment(components, adj, measurement_nodes)
    experiment.checkexperiment()

    experiment.draw(titles = 'both')

    experiment.make_path()
    experiment.check_path()

    # here we initialize what pulse we inject into each starting node
    for node in experiment.nodes():
        if len(experiment.pre(node)) == 0:
            experiment.nodes[node]['input'] = env.At

        
    gap = GeneticAlgorithmParameters()
    gap.NFITNESS = 2
    gap.WEIGHTS = (1.0,0.1)
    gap.MULTIPROC = True
    gap.NCORES = mp.cpu_count()
    gap.N_POPULATION = 50      # number of individuals in a population
    gap.N_GEN = 30              # number of generations
    gap.MUT_PRB = 0.2           # independent probability of mutation
    gap.CRX_PRB = 0.95          # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 1             # verbose print statement for GA statistics
    
    print('Number of cores: {}, number of generations: {}, size of population: {}'.format(gap.NCORES, gap.N_GEN, gap.N_POPULATION))
    
    tstart = time.time()
    hof, population, logbook = inner_geneticalgorithm(gap, env, experiment)
    tstop = time.time()

#    log = extractlogbook(logbook)    
#    plt.figure()
#    plt.plot(log['gen'], log['max'], label='max') 
#    plt.legend()
    
    print('\nElapsed time = {}'.format(tstop-tstart))
    print('Total number of individuals measured: {}\n'.format(sum(log['nevals'])))
#    """
    
    
    
    """
    Now we visualizing the best HOF individual found, and slightly improve it
    """    
    for j in range(gap.N_HOF):
        individual = hof[j]
        measurement_node = experiment.measurement_nodes[0]
        
        print(individual)
        
        experiment.setattributes(individual)
        experiment.simulate(env)
        
        At = experiment.nodes[measurement_node]['output'].reshape(env.N)
        fitness = env.fitness(At)
        print(fitness)
        
        experiment.measure(env, measurement_node)    
        plt.show()
        
        
        # Now fine tune the best of the hall of fame
        individual_fine = finetune_individual(individual, env, experiment)
        
        experiment.setattributes(individual_fine)
        experiment.simulate(env)
        
        At = experiment.nodes[measurement_node]['output'].reshape(env.N)
        fitness = env.fitness(At)
        
        experiment.measure(env, measurement_node)  
        plt.show()
                
        if save:
            save_experiment(filename, experiment, env)
#        
#    return experiment, env

#if __name__ == '__main__': 
#    (experiment, env) = main()
