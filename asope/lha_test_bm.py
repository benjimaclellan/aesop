"""

"""

#%% this allows proper multiprocessing (overrides internal multiprocessing settings)
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

#%% import public modules
import time
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

import seaborn
import multiprocess as mp
import copy 
import autograd.numpy as np
import pandas as pd
from autograd import elementwise_grad, jacobian
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

from fitness_analysis import drop_node, remove_redundancies, mc_error_propagation, compute_moment_matrices, compute_interpolation_points, multivariable_simulate
from fitness_analysis import simulate_with_error, get_error_parameters, get_error_functions, UDR_moment_approximation, UDRAnalysis, UDR_moments

plt.close("all")

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

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
    gap.NFITNESS = 2            # how many values to optimize
    gap.WEIGHTS = (1.0, 1.0)    # weights to put on the multiple fitness values
    gap.MULTIPROC = True        # multiprocess or not
    gap.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gap.N_POPULATION = 200      # number of individuals in a population (make this a multiple of NCORES!)
    gap.N_GEN = 100             # number of generations
    gap.MUT_PRB = 0.5           # independent probability of mutation
    gap.CRX_PRB = 0.5           # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = True          # verbose print statement for GA statistics
    gap.INIT = None
    gap.FINE_TUNE = True
    gap.NUM_ELITE = 1
    gap.NUM_MATE_POOL = gap.N_POPULATION//2 - gap.NUM_ELITE

    #%% initialize our input pulse, with the fitness function too
#    env = OpticalField_Pulse(n_samples=2**14, profile='gaussian', pulse_width=100e-12, f_rep=100e6, n_pulses=15, peak_power=1)
#    env.init_fitness(p=1, q=2)

    env = OpticalField_CW(n_samples=2**13, window_t=10e-9, peak_power=1)
    target_harmonic = 12e9
    env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=False)
    #%%

    components = {
                    0:PhaseModulator(),
                    1:WaveShaper(),
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
    at = hof[0]
    print(at)
    
    exp.setattributes(at)
    exp.simulate(env)

    At = exp.nodes[exp.measurement_nodes[0]]['output']

    At = env.shift_function(exp.nodes[exp.measurement_nodes[0]]['output'])

    fit = env.fitness(At)
    print("Fitness: " + str(fit))

    extent = 1000#env.n_samples
    plt.figure()
    plt.plot(env.t[0:extent], P(At[0:extent]), label='output')
    plt.plot(env.t[0:extent], P(env.At0[0:extent]), label='input')
    plt.plot(env.t[0:extent], env.target[0:extent], label='target')
    plt.legend()
#    plt.title('p = {}, q = {}'.format(env.p, env.q))
    plt.show()


    f = lambda x: multivariable_simulate(x, exp, env)
    print("Beginning Autodifferentiation")
#
    # Compute the Hessian of the fitness function (as a function of x)
    t1 = time.clock()
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

    muv = exp.attributes_to_vector()
    sigma_list = exp.get_sigma_vector()
    H0 = Hf(muv)
    H0 = H0/2 # Taylor exp. factor of 1/2!

    print(H0)

    print("Symmetry Check")

    sym_dif = H0 - H0.T
    print("Max asymmetry " + str(np.amax(sym_dif)))

    t2 = time.clock()
    eigen_items = np.linalg.eig(H0)
    eigensort_inds = np.argsort(np.abs(eigen_items[0]))
    eigenvalues = eigen_items[0][eigensort_inds]
    eigenvectors = eigen_items[1][:,eigensort_inds]
    eig_time = time.clock() - t2
    print('eig_time = {}'.format(eig_time))
    
    fig = plt.figure()
    ax = seaborn.heatmap((H0),linewidths=.5)
    ax.set_aspect("equal")
    ax.set_xticklabels(at_name, rotation=60, ha='right')
    ax.set_yticklabels(at_name, rotation=-30, ha='right', va='bottom')
    fig.tight_layout()
    plt.show()

    
    print("\n", np.linalg.det(H0), np.trace(H0))
        
    xval = np.arange(0,eigenvalues.shape[0],1)
    fig, ax = plt.subplots(np.ceil(eigenvectors.shape[1]/3).astype('int'),3, sharex=True, sharey=True, gridspec_kw = {'wspace':0.2, 'hspace':0})
#    fig = gridspec.GridSpec(4, 4)
    ax = ax.flatten()
    for k in range(0, eigenvectors.shape[1]): 
        ax[k].stem(eigenvectors[:,k], linefmt='teal', markerfmt='o', label = 'Eigenvalue {} = {:1.3e}'.format(k, (eigenvalues[k])))
        ax[k].legend()
    plt.ylabel('Linear Coefficient')
    plt.xlabel('Component Basis') 
    plt.show()
    
    plt.figure()
    xval = np.arange(0,eigenvalues.shape[0],1)
    plt.stem(xval-0.05, ((np.diag(H0))),  linefmt='salmon', markerfmt= 'x', label='Hessian diagonal')
    plt.stem(xval+0.05, (eigenvalues), linefmt='teal', markerfmt='o', label='eigenvalues')
    plt.xticks(xval)
    plt.xlabel('Component Basis')
    plt.ylabel("Value")
    plt.title("Hessian Diagonal")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.stem(xval+0.05, (eigenvalues), linefmt='teal', markerfmt='o', label='eigenvalues', use_line_collection=True)
#    plt.xticks(xval, labels=at_name, rotation=60, ha='right')
    plt.xlabel('Hessian eigenvalues')
    plt.ylabel("Value")
    plt.title("Hessian Spectrum")
    plt.legend()
    plt.tight_layout()
    plt.show()

    exp.visualize(env, at=at)
    plt.show()
    print(at)
    
    #%%
    comp_perms = np.array(list(itertools.combinations(list(exp.nodes()), 2)))
    for perm in comp_perms:
        print(basis, perm)
        rows_cols1, = np.where(basis == perm[0])
        rows_cols2, = np.where(basis == perm[1])
        rows_cols = np.concatenate((rows_cols1, rows_cols2))
        print(rows_cols)
    
        H0_mask = H0[np.ix_(rows_cols, rows_cols)]
        print(H0_mask)
        sub_names = np.array(at_name)[rows_cols]
        
        eigen_items = np.linalg.eig(H0_mask)
        eigensort_inds = np.argsort(np.abs(eigen_items[0]))
        eigenvalues = eigen_items[0][eigensort_inds]
        eigenvectors = eigen_items[1][:,eigensort_inds]
        
        
        fig = plt.figure()
        ax = seaborn.heatmap((H0_mask),linewidths=.5)
        ax.set_aspect("equal")
        ax.set_xticklabels(sub_names, rotation=60, ha='right')
        ax.set_yticklabels(sub_names, rotation=-30, ha='right', va='bottom')
        fig.tight_layout()
        plt.title(basis)
        plt.show()
        
        
        xval = np.arange(0,eigenvalues.shape[0],1)
        fig, ax = plt.subplots(np.ceil(eigenvectors.shape[1]/3).astype('int'),3, sharex=True, sharey=True, gridspec_kw = {'wspace':0.2, 'hspace':0})
        ax = ax.flatten()
        for k in range(0, eigenvectors.shape[1]): 
            ax[k].stem(eigenvectors[:,k], linefmt='teal', markerfmt='o', use_line_collection=True, label = 'Eigenvalue {} = {:1.3e}'.format(k, (eigenvalues[k])))
            ax[k].set_ylabel('{:1.0e}'.format((eigenvalues[k])))
        plt.xlabel('Component Basis') 
        plt.title(perm)
        plt.show()  
        
    multipage('results/2019_07_30__interestingresults102.pdf')

    df = elementwise_grad(f)
    print(df(muv))
    perturb_vec = (sigma_list)
    predicted_fit = f(muv) + np.dot(df(muv), perturb_vec)
    #predicted_fit += np.dot(perturb_vec, np.dot(H0, perturb_vec))
    print("Prediction: " + str(predicted_fit))

    lengths = sigma_list[-1]*np.linspace(-1.5,1.5,200)
    fits = []
    dfits = []
    for item in lengths:
        tmp = muv + [0,0,0,item]
        fits.append(f(tmp))
        dfits.append(df(tmp))

    plt.plot(lengths/sigma_list[-1],fits)
    plt.title("Fitness landscape slice (fiber L)")
    plt.xlabel("Deviation from mean")
    plt.axvline(0)
    plt.ylabel("Fitness")
    plt.show()

    plt.plot(lengths/sigma_list[-1],dfits)
    plt.title("Derivative slice of fit")
    plt.xlabel("Deviation from mean")
    plt.ylabel("Derivative of Fitness")
    plt.show()

    real_fit = f(muv)
    print("RV: " + str(real_fit))
