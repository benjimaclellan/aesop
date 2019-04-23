#%%
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

#import time
import matplotlib.pyplot as plt
import multiprocess as mp

from assets.functions import extractlogbook, save_experiment, load_experiment
from assets.environment import PulseEnvironment

from assets import config

from assets.classes import Experiment, GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.geneticalgorithmouter import outer_geneticalgorithm

from optimization.gradientdescent import finetune_individual

from assets.graph_manipulation import change_experiment_wrapper, brand_new_experiment, remake_experiment

#from asope_main import optimize_experiment

plt.close("all")

#%%
if __name__ == '__main__': 
    
    # store all our hyper-parameters for the genetic algorithm
    gapO = GeneticAlgorithmParameters()
    gapO.TYPE = "outer"
    gapO.NFITNESS = 2            # how many values to optimize
    gapO.WEIGHTS = (1.0, 1.0)   # weights to put on the multiple fitness values
    gapO.MULTIPROC = False       # multiprocess or not
    gapO.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gapO.N_POPULATION = 3       # number of individuals in a population
    gapO.N_GEN = 10               # number of generations
    gapO.MUT_PRB = 1.0           # independent probability of mutation
    gapO.CRX_PRB = 0.0          # independent probability of cross-over
    gapO.N_HOF = 3               # number of inds in Hall of Fame (num to keep)
    gapO.VERBOSE = 1             # verbose print statement for GA statistics
    gapO.INIT = None
    gapO.NUM_ELITE = 2
    gapO.NUM_MATE_POOL = 0
    
    env = PulseEnvironment(p = 3, q = 2, profile = 'sech')    
    hof, population, logbook = outer_geneticalgorithm(gapO, env)
    
    for k in range(0,1):
        experiment = remake_experiment(hof[k])
        experiment.setattributes(hof[k].inner_attributes)
        experiment.draw(node_label = 'both')
        
        measurement_nodes = []
        for node in experiment.nodes():
            if len(experiment.suc(node)) == 0:
                measurement_nodes.append(node)
                
        for node in experiment.nodes():
            if len(experiment.pre(node)) == 0:
                experiment.nodes[node]['input'] = env.At
            
        print(hof[k].inner_attributes)
        
        experiment.simulate(env)
        
        experiment.measure(env, measurement_nodes[0], check_power=True)  
        plt.show()
        
#        save_experiment("results/current", experiment)
    
