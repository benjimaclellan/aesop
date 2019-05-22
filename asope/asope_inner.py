"""
Copyright Benjamin MacLellan

The inner optimization process for the Automated Search for Optical Processing Experiments (ASOPE). This uses a genetic algorithm (GA) to optimize the parameters (attributes) on the components (nodes) in the experiment (graph).

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
from assets.graph_manipulation import get_nonsplitters

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, AmplitudeModulator
from classes.experiment import Experiment
from classes.geneticalgorithmparameters import GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

from noise_sim import update_error_attributes, simulate_component_noise, drop_node, remove_redundancies

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

        #%% Now fine tune the best individual using gradient descent
        if gap.FINE_TUNE:
            print("Fine tuning with gradient descent...")
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
    gap.N_POPULATION = 100      # number of individuals in a population
    gap.N_GEN = 5             # number of generations
    gap.MUT_PRB = 0.5           # independent probability of mutation
    gap.CRX_PRB = 0.5          # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = 0             # verbose print statement for GA statistics
    gap.INIT = None
    gap.FINE_TUNE = True
    gap.NUM_ELITE = 1
    gap.NUM_MATE_POOL = gap.N_POPULATION//2 - gap.NUM_ELITE
    
    #%% initialize our input pulse, with the fitness function too
    env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)
    target_harmonic = 12e9
    #env.createnoise()
    #env.addnoise()
    
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
                    3:Fiber(),
                    4:WaveShaper()
                 }
    adj = [(0,1),(1,2),(2,3),(3,4)]

    #%% initialize the experiment, and perform all the preprocessing steps
    exp = Experiment()
    exp.buildexperiment(components, adj)
    exp.checkexperiment()

    exp.make_path()
    exp.check_path()
    exp.inject_optical_field(env.At)
    
    exp.draw(node_label = 'disp_name')

    
    #%%
    exp, hof, hof_fine, log = optimize_experiment(exp, env, gap, verbose=True)
    
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


    """
    Redundancy Check
    """

    print("Beginning Redundancy Check")
    exp = remove_redundancies(exp, env, gap.VERBOSE)
    plt.show()
    exp.draw()
    plt.show()


    """
    Robustness/Noise Simulation 
    """
    print("Beginning Monte Carlo simulation")
    # Number of Monte Carlo trials to preform
    N_samples = 20

    # Generate an array of fitness
    fitnesses, optical_fields = simulate_component_noise(exp, env, At, N_samples)
    if gap.VERBOSE:
        print("Fitness Array: ")
        print(fitnesses)

    # Calculate statistics (mean/std) of the tests
    i = 0
    for row in fitnesses.T:
        std = np.std(row)
        mean = np.mean(row)
        print("Mean of column " + str(i) + " : " + str(mean))
        print("Standard deviation of column " + str(i) + " : " + str(std))
        i += 1

    print("________________")

    At_avg = np.mean(optical_fields, axis=0)
    At_std = np.std(optical_fields, axis=0)

    # clear memory space
    del At
    del fitnesses
    del optical_fields

    #%%
    plt.figure()
    #generated = env.shift_function(P(At_avg))
    minval, maxval = np.min(At_avg), np.max(At_avg)
    if env.normalize:
        #TODO: Determine how to properly normalize STD
        At_avg = (At_avg-minval)/(maxval-minval)
        At_std = At_std/maxval


    #plt.plot(env.t, generated,label='current')
    plt.plot(env.t, env.target,label='target',ls=':')
    plt.plot(env.t, env.At0, label='initial')
    plt.plot(env.t, At_avg,'r', label='current')
    plt.plot(env.t, At_avg + At_std, 'r--')
    plt.plot(env.t, At_avg - At_std, 'r--')
    plt.xlim([0,10/env.target_harmonic])
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(np.abs(RFSpectrum(env.target, env.dt)),label='target',ls=':')
    plt.plot(np.abs(RFSpectrum(At_avg, env.dt)),label='current')
    plt.plot(np.abs(RFSpectrum(At_avg + At_std, env.dt)), label='upper std')
    plt.plot(np.abs(RFSpectrum(At_avg - At_std, env.dt)), label='lower std')
    plt.legend()
    plt.show()
    
    exp.visualize(env)
    plt.show()