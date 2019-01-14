import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import time
import matplotlib.pyplot as plt
import scipy.optimize as opt
import multiprocess as mp
import numpy as np

from components.component_parameters import component_parameters

from classes import GeneticAlgorithmParameters
from functions.helperfunctions import extract_bounds, experiment_description, buildexperiment, extractlogbook

from environments.environment_pulse import PulseEnvironment
from simulators.simulator_classical import ClassicalSimulator

from geneticalgorithms.ga_functions_inner import inner_geneticalgorithm, FIT_Inner

plt.close("all")

## ************************************************
def final_opt(ind, gap, env, experiment, sim):
    fitness = -FIT_Inner(ind,gap, env, experiment, sim)[1]
    return fitness

if __name__ == '__main__':  
    
    """   
    ASOPE V1 is a program which creates an optimal experimental setup for a desired outcome. The software will generate a random setup from the available component, and optimize parameters
    TODO:
        Implement these, such that the next run includes one individual that uses the best values from the previous run
            USE_PREVIOUS_HOF  = False
            SAVE_PREVIOUS_HOF = False
    """     
    

    fitnessfunction = 'TemporalTalbot'
    sim_kwargs = {'p':5, 'q':1}

    env = PulseEnvironment()
    sim = ClassicalSimulator(fitnessfunction, **sim_kwargs)

    experiment_nums = [4, 0]
    component_parameters = component_parameters(env, sim)
    experiment = buildexperiment(component_parameters, experiment_nums)

    (N_ATTRIBUTES, BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS) = extract_bounds(experiment)
    
    gap = GeneticAlgorithmParameters(N_ATTRIBUTES, BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS)
    
    gap.NFITNESS = 2
    gap.WEIGHTS = (1.0, 1.0)
    gap.MULTIPROC = False
    gap.NCORES = mp.cpu_count()
    gap.N_POPULATION = 50      # number of individuals in a population
    gap.N_GEN = 20              # number of generations
    gap.MUT_PRB = 0.2           # independent probability of mutation
    gap.CRX_PRB = 0.2           # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 1             # verbose print statement for GA statistics
    
    tstart = time.time()
    hof, population, logbook = inner_geneticalgorithm(gap, env, experiment, sim)
    tstop = time.time()

    log = extractlogbook(logbook)    
    
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
        res = opt.minimize(final_opt, individual, args=(gap, env, experiment, sim), method='CG', options={'maxiter':100})        
        individual =  list(res.x)
        print(individual)
        
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
