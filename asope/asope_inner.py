"""
Copyright Benjamin MacLellan

The inner optimization process for the Automated Search for Optical Processing Experiments (ASOPE). This uses a genetic algorithm (GA) to optimize the parameters (attributes) on the components (nodes) in the experiment (graph).

"""

#%% this allows proper multiprocessing (overrides internal multiprocessing settings)
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import warnings
warnings.filterwarnings("ignore")

#%% import public modules
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocess as mp
import copy 
import autograd.numpy as np
import pandas as pd
from scipy import signal
import seaborn

#%% import custom modules
from assets.functions import extractlogbook, save_class, load_class, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

from assets.waveforms import random_bit_pattern, bit_pattern, rf_chirp
from assets.callbacks import save_experiment_and_plot

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

        #%% Now fine tune the best individual using gradient descent
        if gap.FINE_TUNE:
            if verbose:
                print('Fine-tuning the most fit individual using quasi-Newton method')

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
    gap.NFITNESS = 1            # how many values to optimize
    gap.WEIGHTS = (1.0,)    # weights to put on the multiple fitness values
    gap.MULTIPROC = True        # multiprocess or not
    gap.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gap.N_POPULATION = 200       # number of individuals in a population (make this a multiple of NCORES!)
    gap.N_GEN = 50               # number of generations
    gap.MUT_PRB = 0.5           # independent probability of mutation
    gap.CRX_PRB = 0.5           # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = True          # verbose print statement for GA statistics
    gap.INIT = None
    gap.FINE_TUNE = True
    gap.NUM_ELITE = 1
    gap.NUM_MATE_POOL = gap.N_POPULATION//2 - gap.NUM_ELITE

    #%% initialize our input pulse, with the fitness function too
#    env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)
#    target_harmonic = 12e9
#    env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=False)

    env = OpticalField_Pulse(n_samples=2**14, profile='gaussian', pulse_width=100e-12, f_rep=100e6, n_pulses=15, peak_power=1)
    env.init_fitness(p=1, q=2)

    
    #%%
    components = {
                    0:Fiber(),
                    1:Fiber(),
                    2:PhaseModulator()
                 }
    adj = [(0,1), (1,2)]

    #%% initialize the experiment, and perform all the pre-processing steps
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
#    fig_log, ax_log = plt.subplots(1,1, figsize=[8,6])
#    ax_log.plot(log['gen'], log["Best [fitness, variance]"], label='Maximum', ls='-', color='salmon', alpha=1.0)
#    ax_log.plot(log['gen'], log["Average [fitness, variance]"], label='Mean', ls='-.', color='blue', alpha=0.7)
#    ax_log.legend()
#    ax_log.set_xlabel('Generation')
#    ax_log.set_ylabel(r'Fitness, $\mathcal{F}(\mathbf{x})$')
#    
    #%%
    at = copy.deepcopy(hof_fine[0])

#    print(at)
    
    exp.setattributes(at)
    exp.simulate(env)
    At = exp.nodes[exp.measurement_nodes[0]]['output']
    fit = env.fitness(At)
    print("Fitness: {}".format(fit))

    plt.figure()
    plt.plot(env.t, P(At))
    plt.plot(env.t, P(env.At0))
    plt.show()

    #%% Redundancy check
#    print("Beginning Redundancy Check")
#    exp1 = remove_redundancies(exp, env, verbose=True)
#    exp1.draw(node_label = 'disp_name')
#    plt.show()

    #%% UDR
#    std_udr = analysis_udr(at, exp, env, verbose=True)
#    
#    ## Monte Carlo
#    mu_mc, std_mc = analysis_mc(at, exp, env, 10**3, verbose=True)
#
#    # LHA
#    H0, eigenvalues, eigenvectors, basis_names = analysis_lha(at, exp, env, verbose=True)
#    
    #%%
#    plt.figure()
#    plt.stem(np.diag(H0)/np.max(np.abs(np.diag(H0))),label='lha')
#    plt.stem(std_udr/np.max(std_udr), label='udr', linefmt='-gx')
#    plt.legend()
#    plt.show()
#    
#    save_class('testing/experiment_example', exp)
#    save_class('testing/environment_example', env)
    
    
    
##    ## PLOTTING FROM HERE ON
#    plt.figure()
#    g = seaborn.heatmap((H0))
#    g.set_xticklabels(basis_names[1], rotation=30)
#    g.set_yticklabels(basis_names[1], rotation=60)
#
#
#    plt.figure()
#    plt.plot(env.t, P(At))
#    plt.plot(env.t, P(env.At0))
#    plt.show()


#    fig, ax = plt.subplots(eigenvectors.shape[1], 1, sharex=True, sharey=True)
#    for k in range(0, eigenvectors.shape[1]):
#        ax[k].stem(eigenvectors[:,k], linefmt='teal', markerfmt='o', label = 'Eigenvalue {} = {:1.3e}'.format(k, (eigenvalues[k])))
#        ax[k].legend()
#    plt.ylabel('Linear Coefficient')
#    plt.xlabel('Component Basis')
#    plt.xticks([j for j in range(0,eigenvectors.shape[0])], labels=at_name)
#
#    stop = time.time()
#    print("T: " + str(stop-start))
#
#    plt.figure()
#    xval = np.arange(0,eigenvalues.shape[0],1)
#    plt.stem(xval-0.05, ((np.diag(H0))),  linefmt='salmon', markerfmt= 'x', label='Hessian diagonal')
#    plt.stem(xval+0.05, (eigenvalues), linefmt='teal', markerfmt='o', label='eigenvalues')
#    plt.xticks(xval)
#    plt.xlabel('Component Basis')
#    plt.ylabel("Value")
#    plt.title("Hessian Spectrum")
#    plt.legend()
#    plt.show()

