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

from assets.waveforms import random_bit_pattern, bit_pattern, rf_chirp
from assets.callbacks import save_experiment_and_plot
from assets.graph_manipulation import get_nonsplitters

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, AmplitudeModulator
from classes.experiment import Experiment
from classes.geneticalgorithmparameters import GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

from noise_sim import update_error_attributes, simulate_component_noise, drop_node, remove_redundancies, UDR_moments, UDR_evCalculation
from noise_sim import simulate_with_error, get_error_parameters, get_error_functions, compute_moment_matrices, compute_interpolation_points

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
    gap.NFITNESS = 1           # how many values to optimize
    gap.WEIGHTS = (1.0),     # weights to put on the multiple fitness values
    gap.MULTIPROC = True        # multiprocess or not
    gap.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gap.N_POPULATION = 100      # number of individuals in a population
    gap.N_GEN = 10             # number of generations
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
    
    env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=False)
    
    #%%
    components = {
                    0:PhaseModulator(),
                    1:WaveShaper()
                 }
    adj = [(0,1)]

    #%% initialize the experiment, and perform all the preprocessing steps
    exp = Experiment()
    exp.buildexperiment(components, adj)
    exp.checkexperiment()

    exp.make_path()
    exp.check_path()
    exp.inject_optical_field(env.At)
    
    exp.draw(node_label = 'disp_name')

    
    #%%
    #exp, hof, hof_fine, log = optimize_experiment(exp, env, gap, verbose=True)
    
    #%%
    #fig_log, ax_log = plt.subplots(1,1, figsize=[8,6])
    #ax_log.plot(log['gen'], log['max'], label='Maximum', ls='-', color='salmon', alpha=1.0)
    #ax_log.plot(log['gen'], log['avg'], label='Mean', ls='-.', color='blue', alpha=0.7)
    #ax_log.legend()
    #ax_log.set_xlabel('Generation')
    #ax_log.set_ylabel(r'Fitness, $\mathcal{F}(\mathbf{x})$')
    
    #%%
    #at = hof_fine[0]
    at = {0: [1.0885386831780766, 10000000000.0], 1: [0.1600913131373453, 0.9644562615852816, 0.8162365069799228, 0.7571936468447649, 0.40455545122113784, 0.0, 0.45345425331484, 6.283185307179586, 4.99676751969036, 3.064033992879826, 2.4118305095380235, 0.4892691534724825, 5.294011788726437, 6.282336557917184]}

    print(at)
    
    exp.setattributes(at)
    exp.simulate(env)

    At = exp.nodes[exp.measurement_nodes[0]]['output']

    fit = env.fitness(At)
    print("Fitness: " + str(fit))

    plt.show()
    plt.plot(env.t, env.target,label='target',ls=':')
    plt.plot(env.t, np.abs(At))
    plt.xlim([0,10/env.target_harmonic])
    plt.show()

    """
    Redundancy Check
    """

    #print("Beginning Redundancy Check")
    #exp = remove_redundancies(exp, env, gap.VERBOSE)
    #plt.show()
    #exp.draw()
    #plt.show()

    """
    Robustness/Noise Simulation 
    """
    print("Beginning Monte Carlo simulation")
    # Number of Monte Carlo trials to preform
    start = time.time()
    N_samples = 500

    # Generate an array of fitness
    fitnesses = simulate_component_noise(exp, env, At, N_samples)
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


    print("sim count " + str(simulate_with_error.count))

    stop = time.time()
    print("Time: " + str(stop - start))
    print("________________")
    print("Beginning Univariate Dimension Reduction")
    start = time.time()
    error_params = get_error_parameters(exp)
    error_functions = get_error_functions(exp)
    # Create a lambda function as we do not want to pass exp,env
    f2 = lambda x: simulate_with_error(x, exp, env) - fit[0]

    matrix_moments = compute_moment_matrices(error_params, error_functions, 5)

    x, r = compute_interpolation_points(matrix_moments)
    xim = np.imag(x)
    xre = np.real(x)
    if np.any(np.imag(x) != 0):
        # check for machine accuracy errors causing imaginary vals
        raise np.linalg.LinAlgError
    x = np.real(x)
    mean = UDR_moments(f2, 1, error_params, error_functions, [x,r], matrix_moments) + fit[0]
    print("mu: " + str(mean))
    #std = np.sqrt(UDR_moments(f2, 2, error_params, error_functions, [x,r], matrix_moments))
    #print("std: " + str(std))

    stop = time.time()
    print("Time: " + str(stop - start))
    print("________________")
    print("sim count " + str(simulate_with_error.count))
    print("udr count " + str(UDR_evCalculation.count))
    print("number of variables " + str(np.shape(error_params)[0]))
    #At_avg = np.mean(np.abs(optical_fields), axis=0)
    #At_std = np.std(np.abs(optical_fields), axis=0)

    #At_avg = env.shift_function(At_avg)

    #noise_sample = np.abs(optical_fields[0])
    #print("Power Check: " + str(exp.power_check_single(At_avg)))
    # clear memory space
    del fitnesses
    #del optical_fields

    #%%
    plt.figure()
    #generated = env.shift_function(P(At_avg))
    #minval, maxval = np.min(At_avg), np.max(At_avg)
    print("Normalize? " + str(env.normalize))
    '''if env.normalize:
        #TODO: Determine how to properly normalize STD
        At_avg = At_avg/maxval
        #At_avg = (At_avg-minval)/(maxval-minval)
        #At_std = At_std/maxval
        noise_sample = (noise_sample-minval)/(maxval-minval)
    '''

    #plt.plot(env.t, generated,label='current')
    #plt.plot(env.t, env.target,label='target',ls=':')
    #plt.plot(env.t, env.At0, label='initial')
    #plt.plot(env.t, At_avg,'r', label='current')
    #plt.plot(env.t, At_avg + At_std, 'r--')
    #plt.plot(env.t, At_avg - At_std, 'r--')
    #plt.plot(env.t, noise_sample, 'k--', label="noise sample")
    #plt.xlim([0,10/env.target_harmonic])
    #plt.legend()
    #plt.show()
    
    #plt.figure()
    #plt.plot(np.abs(RFSpectrum(env.target, env.dt)),label='target',ls=':')
    #plt.plot(np.abs(RFSpectrum(At_avg, env.dt)),label='current')
    #plt.plot(np.abs(RFSpectrum(At_avg + At_std, env.dt)), label='upper std')
    #plt.plot(np.abs(RFSpectrum(At_avg - At_std, env.dt)), label='lower std')
    #plt.legend()
    #plt.show()

    #exp.visualize(env)
    #plt.show()

#    save_experiment_and_plot(exp, env, At)

