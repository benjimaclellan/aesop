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
import autograd.numpy as np
import pandas as pd
from autograd import elementwise_grad, jacobian
from scipy import signal
import seaborn

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

from noise_sim import drop_node, remove_redundancies, mc_error_propagation, compute_moment_matrices, compute_interpolation_points, multivariable_simulate
from noise_sim import simulate_with_error, get_error_parameters, get_error_functions, UDR_moment_approximation, UDRAnalysis, UDR_moments

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

def autograd_hessian(fun, argnum = 0):
    '''
    Compute the hessian by computing the transpose of the jacobian of the gradient.

    :param fun:
    :param argnum:
    :return:
    '''

    def sum_latter_dims(x):
        return np.sum(x.reshape(x.shape[0], -1), 1)

    def sum_grad_output(*args, **kwargs):
        return sum_latter_dims(elementwise_grad(fun)(*args, **kwargs))

    return jacobian(sum_grad_output, argnum)
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
    fig_log, ax_log = plt.subplots(1,1, figsize=[8,6])
    ax_log.plot(log['gen'], log["Best [fitness, variance]"], label='Maximum', ls='-', color='salmon', alpha=1.0)
    ax_log.plot(log['gen'], log["Average [fitness, variance]"], label='Mean', ls='-.', color='blue', alpha=0.7)
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

    #%%
    plt.figure()
    generated = env.shift_function(P(At))
    minval, maxval = np.min(At), np.max(At)
    print("Normalize? " + str(env.normalize))
    if env.normalize:
        #TODO: Determine how to properly normalize STD
        At_norm = At / maxval
        #At_avg = (At_avg-minval)/(maxval-minval)
        #At_std = At_std/maxval

    ## UDR
    '''
    print("Beginning Univariate Dimension Reduction")

    print("Output standard deviation varying individual parameters:")
    stds = np.sqrt(UDRAnalysis(exp, env))
    comp_labels = []
    at_labels = []
    for node, key in get_error_parameters(exp):
        # Get the component labels and construct a string for the dataframe
        node, key = int(node), int(key)
        try:
            title = exp.nodes()[node]['title']
            label = exp.nodes()[node]['info'].at_labels[key]
            comp_labels.append(title)
            at_labels.append(label)
        except AttributeError or TypeError:
            print("Error getting component and attribute labels")

    results = pd.DataFrame(comp_labels, columns=["Component"])
    results = results.assign(Attribute = at_labels)
    results = results.assign(Output_Deviation = stds)
    print(results)
    simulate_with_error.count = 0

    ## Compute the interpolation points
    error_params = get_error_parameters(exp)
    error_functions = get_error_functions(exp)
    f = lambda x: simulate_with_error(x, exp, env) - fit[0]
    matrix_moments = compute_moment_matrices(error_params, 5)
    x, r = compute_interpolation_points(matrix_moments)

    ## Make sure there wasn't any underflow errors etc
    xim = np.imag(x)
    xre = np.real(x)
    if np.any(np.imag(x) != 0):
        raise np.linalg.LinAlgError("Complex values found in interpolation points")
    x = np.real(x)

    ## Compute moments of the output distribution
    simulate_with_error.count = 0
    start = time.time()
    mean = UDR_moments(f, 1, error_params, error_functions, [x,r], matrix_moments) + fit[0]

    print("Results for all variables together")
    print("Mean fitness: " + str(mean))
    std = np.sqrt(UDR_moments(f, 2, error_params, error_functions, [x,r], matrix_moments))
    print("Standard Deviation: " + str(std))
    stop = time.time()
    print("Time elapsed: " + str(stop - start))
    #print("number of function calls : " + str(simulate_with_error.count))
    print("________________")

    #plt.figure()
    #plt.plot(np.abs(RFSpectrum(env.target, env.dt)),label='target',ls=':')
    #plt.plot(np.abs(RFSpectrum(At_avg, env.dt)),label='current')
    #plt.plot(np.abs(RFSpectrum(At_avg + At_std, env.dt)), label='upper std')
    #plt.show()

    exp.visualize(env)
    plt.show()

    ## Monte Carlo
    print("Beginning Monte Carlo Simulation")
    N = 1000
    start = time.time()
    fitnesses = mc_error_propagation(exp, env, N)
    stop = time.time()

    mu = np.mean(fitnesses)
    std = np.std(fitnesses)

    print("Time elapsed: " + str(stop-start) + "s")
    print('Mean fitness ' + str(mu))
    print('Standard deviation ' + str(std))
    # make histogram
    num_bins = 20
    n, bins, patches = plt.hist(fitnesses, num_bins)
    plt.title("Output distribution")
    plt.show()
    '''
    f = lambda x: multivariable_simulate(x, exp, env)
    print("Beginning Autodifferentiation")

    # Compute the Hessian of the fitness function (as a function of x)
    start = time.time()
    Hf = autograd_hessian(f)

    # Construct a vector of the mean value, and a vector of the standard deviations.
    muv, sigma_list, basis, at_name = [], [], [], []
    j,k = (0, 0)
    for node in exp.nodes():
        for name in exp.nodes[node]['info'].AT_VARS:
            at_name.append('{}:{}'.format(node,name))
        for q in at[node]:
            muv.append(q)
            basis.append(node)
            j += 1
        for mu, sigma in exp.nodes[node]['info'].at_pdfs:
            sigma_list.append(sigma)
            k += 1

    muv, sigma_list, basis = np.array(muv), np.array(sigma_list), np.array(basis)

    print(muv)
    H0 = Hf(muv)
    H0 = H0/2 # Taylor exp. factor of 1/2!
    i = 0
    for row in H0:
        j = 0
        for val in row:
            sigma_i = sigma_list[i]
            sigma_j = sigma_list[j]
            H0[i, j] = val*sigma_i*sigma_j
            j += 1
        i += 1

    print(H0)

    print("Symmetry Check")

    sym_dif = H0 - H0.T
    print("Max asymmetry " + str(np.amax(sym_dif)))

    eigen_items = np.linalg.eig(H0)
    eigensort_inds = np.argsort(eigen_items[0])
    eigenvalues = eigen_items[0][eigensort_inds]
    eigenvectors = eigen_items[1][:,eigensort_inds]

    plt.figure()
    g = seaborn.heatmap((H0))
    g.set_xticklabels(at_name, rotation=0)
    g.set_yticklabels(at_name, rotation=90)

    fig, ax = plt.subplots(eigenvectors.shape[1], 1, sharex=True, sharey=True)
    for k in range(0, eigenvectors.shape[1]):
        ax[k].stem(eigenvectors[:,k], linefmt='teal', markerfmt='o', label = 'Eigenvalue {} = {:1.3e}'.format(k, (eigenvalues[k])))
        ax[k].legend()
    plt.ylabel('Linear Coefficient')
    plt.xlabel('Component Basis')
    plt.xticks([j for j in range(0,eigenvectors.shape[0])], labels=at_name)

    stop = time.time()
    print("T: " + str(stop-start))

    plt.figure()
    xval = np.arange(0,eigenvalues.shape[0],1)
    plt.stem(xval-0.05, ((np.diag(H0))),  linefmt='salmon', markerfmt= 'x', label='Hessian diagonal')
    plt.stem(xval+0.05, (eigenvalues), linefmt='teal', markerfmt='o', label='eigenvalues')
    plt.xticks(xval)
    plt.xlabel('Component Basis')
    plt.ylabel("Value")
    plt.title("Hessian Spectrum")
    plt.legend()
    plt.show()

    perturbed_vector = eigenvectors[0]
    exp.attributes_from_vector(perturbed_vector)
    exp.simulate(env)

    At = exp.nodes[exp.measurement_nodes[0]]['output']

    fit = env.fitness(At)
    print("Fitness: " + str(fit))

    print(np.dot(H0, perturbed_vector))
    # TODO: Figure out why this dot product isn't giving expected val