"""
Copyright Benjamin MacLellan

The inner optimization process for the Automoated Search for Optical Processing Experiments (ASOPE). This uses a genetic algorithm (GA) to optimize the parameters (attributes) on the components (nodes) in the experiment (graph). 

"""

#%% this allows proper multiprocessing (overrides internal multiprocessing settings)
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

#%% import public modules
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocess as mp
import copy 
import numpy as np
from scipy import signal 

#%% import custom modules
from assets.functions import extractlogbook, save_experiment, load_experiment, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.waveforms import random_bit_pattern

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, AmplitudeModulator
from classes.experiment import Experiment
from classes.geneticalgorithmparameters import GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

#%%
def optimize_experiment(experiment, env, gap, verbose=False): 
    
    if verbose:
        print('Number of cores: {}, number of generations: {}, size of population: {}'.format(gap.NCORES, gap.N_GEN, gap.N_POPULATION))
    
    # run (and time) the genetic algorithm
    tstart = time.time()
    hof, population, logbook = inner_geneticalgorithm(gap, env, experiment)
    tstop = time.time()
    
    if verbose:
        print('\nElapsed time = {}'.format(tstop-tstart))
    
    #%% convert DEAP logbook to easier datatype
    log = extractlogbook(logbook)    
           
    #%%    
    hof_fine = []
    for j in range(gap.N_HOF):
        individual = copy.deepcopy(hof[j])        
        hof_fine.append(individual)

        #%% Now fine tune the best individual using grad`ient descent
        if gap.FINE_TUNE:
            individual_fine = finetune_individual(individual, env, experiment)
        else:
            individual_fine = individual
        hof_fine.append(individual_fine)

    return experiment, hof, hof_fine, log

#%%
if __name__ == '__main__': 

    #%% store all our hyper-parameters for the genetic algorithm
    gap = GeneticAlgorithmParameters()
    gap.TYPE = "inner"
    gap.NFITNESS = 1           # how many values to optimize
    gap.WEIGHTS = (1.0),     # weights to put on the multiple fitness values
    gap.MULTIPROC = True        # multiprocess or not
    gap.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gap.N_POPULATION = 200      # number of individuals in a population
    gap.N_GEN = 200              # number of generations
    gap.MUT_PRB = 0.5           # independent probability of mutation
    gap.CRX_PRB = 0.5          # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 1             # verbose print statement for GA statistics
    gap.INIT = None
    gap.FINE_TUNE = True
    gap.NUM_ELITE = 1
    gap.NUM_MATE_POOL = gap.N_POPULATION//2 - gap.NUM_ELITE
    
    #%% initialize our input pulse, with the fitness function too
    env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)    
    target_harmonic = 12e9
    
    env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.25)+1), target_harmonic, normalize=True)
    
#    target, bit_sequence = random_bit_pattern(env.n_samples, 8, target_harmonic, 3/8, env.dt)
#    env.init_fitness(target, target_harmonic, normalize=True)
#    env.bit_sequence = bit_sequence
#    print(bit_sequence)
    
    #%%
    components = {
                    0:PhaseModulator(),
                    1:WaveShaper(),
                    2:PhaseModulator(),
                    3:WaveShaper(),
                 }
    adj = [(0,1), (1,2), (2,3)]

    #%% initialize the experiment, and perform all the preprocessing steps
    exp = Experiment()
    exp.buildexperiment(components, adj)
    exp.checkexperiment()

    exp.make_path()
    exp.check_path()
    exp.inject_optical_field(env.At)
    
    exp.draw(node_label = 'disp_name')

    
    #%%
    exp, hof, hof_fine, log = optimize_experiment(exp, env, gap, verbose=False) 
    
    #%%
    fig_log, ax_log = plt.subplots(1,1, figsize=[8,6])
    ax_log.plot(log['gen'], log['max'], label='Maximum', ls='-', color='salmon', alpha=1.0)
    ax_log.plot(log['gen'], log['avg'], label='Mean', ls='-.', color='blue', alpha=0.7)
    ax_log.legend()
    ax_log.set_xlabel('Generation')
    ax_log.set_ylabel(r'Fitness, $\mathcal{F}(\mathbf{x})$')
    
    #%%
    at = hof_fine[0]
    print(at)
    
    exp.setattributes(at)
    exp.simulate(env)
    
    At = exp.nodes[exp.measurement_nodes[0]]['output']
    
    fit = env.fitness(At)
    print(fit)
    
    #%%
    plt.figure()
    generated = env.shift_function(P(At))
    minval, maxval = np.min(generated), np.max(generated)
    if env.normalize:
        generated = (generated-minval)/(maxval-minval)

    plt.plot(env.t, generated,label='current')
    plt.plot(env.t, env.target,label='target',ls=':')
    plt.xlim([0,10/env.target_harmonic])
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(RFSpectrum(env.target, env.dt),label='target',ls=':')
    plt.plot(RFSpectrum(At, env.dt),label='current')
    plt.legend()
    plt.show()
    
    exp.visualize(env)
    plt.show()