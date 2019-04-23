
"""
The main optimization script for ASOPE, which uses a genetic algorihtm (GA) to optimize the parameters (attributes) on the components (nodes) in the experiment (graph). 
"""
"""
##BUGS 
- import/export control

"""
#%%
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocess as mp
import uuid
import copy 
import scipy

from assets.functions import extractlogbook, save_experiment, load_experiment
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from assets.classes import Experiment, GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

#%%
def optimize_experiment(experiment, env, vis=False): 
    # store all our hyper-parameters for the genetic algorithm
    gap = GeneticAlgorithmParameters()
    gap.TYPE = "inner"
    gap.NFITNESS = 2           # how many values to optimize
    gap.WEIGHTS = (1.0, 1.0)     # weights to put on the multiple fitness values
    gap.MULTIPROC = True        # multiprocess or not
    gap.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gap.N_POPULATION = 100      # number of individuals in a population
    gap.N_GEN = 30              # number of generations
    gap.MUT_PRB = 0.2           # independent probability of mutation
    gap.CRX_PRB = 0.95          # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 0             # verbose print statement for GA statistics
    gap.INIT = None
    gap.NUM_ELITE = 5
    gap.NUM_MATE_POOL = 10
    
    if vis:
        print('Number of cores: {}, number of generations: {}, size of population: {}'.format(gap.NCORES, gap.N_GEN, gap.N_POPULATION))
    
    # run (and time) the genetic algorithm
    tstart = time.time()
    hof, population, logbook = inner_geneticalgorithm(gap, env, experiment)
    tstop = time.time()
    
    if vis:
        print('\nElapsed time = {}'.format(tstop-tstart))
    
    #%%

    if vis and logbook != None: 
        log = extractlogbook(logbook)    
        print('Total number of individuals measured: {}\n'.format(sum(log['nevals'])))
        fig_log, ax_log = plt.subplots(1,1, figsize=[8,6])
        ax_log.plot(log['gen'], log['max'], label='Maximum', ls='-', color='salmon', alpha=1.0)
        ax_log.plot(log['gen'], log['avg'], label='Mean', ls='-.', color='blue', alpha=0.7)
        ax_log.plot(log['gen'], log['std'], label='St. dev.', ls=':', color='orange', alpha=0.4)
        ax_log.plot(log['gen'], log['min'], label='Minimum', ls='-', color='grey', alpha=0.3)
        ax_log.legend()
        ax_log.set_xlabel('Generation')
        ax_log.set_ylabel(r'Fitness, $\mathcal{F}(\mathbf{x})$')
       
    
    #%%
    # now we visualizing the best HOF individual found, and slightly improve it
    
    hof_fine, hof_fine_fitness = [], []
    
    for j in range(gap.N_HOF):
        individual = copy.deepcopy(hof[j])
        measurement_node = experiment.measurement_nodes[0]            
        
        # run experiment with the best individual of the optimization
        experiment.setattributes(individual)
        experiment.simulate(env)
        
        At = experiment.nodes[measurement_node]['output'].reshape(env.N)
        fitness = env.fitness(At)
        
        if vis:
            print(individual)
            print(fitness)
            experiment.measure(env, measurement_node)    
            plt.show()
            
        # Now fine tune the best individual using grad`ient descent
        individual_fine = finetune_individual(individual, env, experiment)
        hof_fine.append(individual_fine)
        
        
        
        experiment.setattributes(individual_fine)
        experiment.simulate(env)
        
        At = experiment.nodes[measurement_node]['output'].reshape(env.N)
        fitness = env.fitness(At)
        hof_fine_fitness.append(fitness)
        
        if vis:
            print(individual_fine)
            experiment.measure(env, measurement_node, check_power=True)  
            plt.show()
                
#        if save:
#            save_experiment(filename, experiment)
        
    return experiment, hof, hof_fine, hof_fine_fitness


if __name__ == '__main__': 
#    filename, save = None, True
##    filename = 'results/' + time.strftime("%Y_%m_%d-%H_%M_%S")
#    filename = 'results/current'
#    if filename == None: 
#        save = False
    
    # initialize our input pulse, with the fitness function too
    env = PulseEnvironment(p = 2, q = 1, profile = 'gauss')

    components = {
    0:AWG(),
    1:Fiber()
    }
    adj = [(0, 1)]
    measurement_nodes = [1]

    # initialize the experiment, and perform all the preprocessing steps
    experiment = Experiment()
    experiment.buildexperiment(components, adj, measurement_nodes)
    experiment.checkexperiment()

    experiment.draw(node_label = 'disp_name')

    experiment.make_path()
    experiment.check_path()

    # here we initialize what pulse we inject into each starting node
    for node in experiment.nodes():
        if len(experiment.pre(node)) == 0:
            experiment.nodes[node]['input'] = env.At
          
    experiment, hof, hof_fine, hof_fine_fitness = optimize_experiment(experiment, env, vis=True) 
