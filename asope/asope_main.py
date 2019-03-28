import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import time
import matplotlib.pyplot as plt
import multiprocess as mp
import uuid
import copy 

from assets.functions import extractlogbook, save_experiment, load_experiment
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from assets.classes import Experiment, GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

"""
The main optimization script for ASOPE, which uses a genetic algorihtm (GA) to optimize the parameters (attributes) on the components (nodes) in the experiment (graph). 
"""

## ************************************************

if __name__ == '__main__': 
    filename, save = None, True
#    filename = 'results/' + time.strftime("%Y_%m_%d-%H_%M_%S")
    filename = 'results/current'
    if filename == None: 
        save = False
    
    # initialize our input pulse, with the fitness function too
    env = PulseEnvironment(p = 3, q = 1, profile = 'gauss')

    components = {
        6:WaveShaper(),
        0:PowerSplitter(),
        5:PowerSplitter(),
        1:Fiber(),
        4:WaveShaper(),
        3:Fiber(),
        2:AWG(),
        }
    adj = [(0, 1), (0, 4), (5, 6), (1, 2), (4, 5), (3, 5), (2, 3)]
    measurement_nodes=[6]
    
#    components = (
#        {
#         'fiber0': Fiber(),
#         'ps0':FrequencySplitter(),
#         'pm0':PhaseModulator(),
#         'fiber1':Fiber(),
#         'ps1':FrequencySplitter(),
#         'fiber2':Fiber(),
#        }
#        ) 
#    adj = [('fiber0','ps0'), ('ps0','pm0'), ('pm0','ps1'), ('ps0','fiber1'), ('fiber1','ps1'), ('ps1', 'fiber2')]
#    measurement_nodes = ['fiber2']
   
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

    # store all our hyper-parameters for the genetic algorithm
    gap = GeneticAlgorithmParameters()
    gap.NFITNESS = 2            # how many values to optimize
    gap.WEIGHTS = (1.0, 1.0)     # weights to put on the multiple fitness values
    gap.MULTIPROC = True        # multiprocess or not
    gap.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gap.N_POPULATION = 300      # number of individuals in a population
    gap.N_GEN = 10              # number of generations
    gap.MUT_PRB = 0.2           # independent probability of mutation
    gap.CRX_PRB = 0.95          # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 1             # verbose print statement for GA statistics
    gap.INIT = None
    
    print('Number of cores: {}, number of generations: {}, size of population: {}'.format(gap.NCORES, gap.N_GEN, gap.N_POPULATION))
    
    # run (and time) the genetic algorithm
    tstart = time.time()
    hof, population, logbook = inner_geneticalgorithm(gap, env, experiment)
    tstop = time.time()
    print('\nElapsed time = {}'.format(tstop-tstart))
    
    # extract the log of the iterations
    if logbook != None: 
        log = extractlogbook(logbook)    
        print('Total number of individuals measured: {}\n'.format(sum(log['nevals'])))
        fig_log = plt.figure(figsize=[8,8], dpi=90)
        plt.plot(log['gen'], log['max'], label='Max', ls='-', color='salmon', alpha=1.0)
        plt.plot(log['gen'], log['avg'], label='Avg', ls='-.', color='blue', alpha=0.7)
        plt.plot(log['gen'], log['std'], label='STD', ls=':', color='orange', alpha=0.4)
        plt.plot(log['gen'], log['min'], label='Min', ls='-', color='grey', alpha=0.3)
        
    # now we visualizing the best HOF individual found, and slightly improve it   
    for j in range(gap.N_HOF):
        individual = copy.deepcopy(hof[j])
#        individual = {0: [3, -3.3231200,0,-3.3231200], 1: [1591.5457908747617]}
        measurement_node = experiment.measurement_nodes[0]
        
        print(individual)
        
        # run experiment with the best individual of the optimization
        experiment.setattributes(individual)
        experiment.simulate(env)
        
        At = experiment.nodes[measurement_node]['output'].reshape(env.N)
        fitness = env.fitness(At)
        print(fitness)
        
        experiment.measure(env, measurement_node)    
        plt.show()
        
        
        # Now fine tune the best individual using grad`ient descent
        individual_fine = finetune_individual(individual, env, experiment)
        print(individual_fine)
        
        experiment.setattributes(individual_fine)
        experiment.simulate(env)
        
        At = experiment.nodes[measurement_node]['output'].reshape(env.N)
        fitness = env.fitness(At)
        
        experiment.measure(env, measurement_node)  
        plt.show()
                
#        if save:
#            save_experiment(filename, experiment)
