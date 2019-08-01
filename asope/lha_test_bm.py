"""

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

from noise_sim import drop_node, remove_redundancies, mc_error_propagation, compute_moment_matrices, compute_interpolation_points, multivariable_simulate
from noise_sim import simulate_with_error, get_error_parameters, get_error_functions, UDR_moment_approximation, UDRAnalysis, UDR_moments

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
    env = OpticalField_Pulse(n_samples=2**13, profile='gaussian', pulse_width=100e-12, f_rep=100e6, n_pulses=15, peak_power=1)
    
    env.init_fitness(p=1, q=2)
    
    #%%
    components = {
                    0:PhaseModulator(),
                    1:Fiber(),
                    2:Fiber(),
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
    at = hof[0]
    print(at)
    
    exp.setattributes(at)
    exp.simulate(env)

    At = exp.nodes[exp.measurement_nodes[0]]['output']

    fit = env.fitness(At)
    print("Fitness: " + str(fit))

    plt.figure()
    plt.plot(env.t, P(At), label='Input')
    plt.plot(env.t, P(env.At0), label='Output')
    plt.legend()
    plt.title('p = {}, q = {}'.format(env.p, env.q))
    plt.show()

    f = lambda x: multivariable_simulate(x, exp, env)
    print("Beginning Autodifferentiation")
#
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

#    multipage('results/2019_07_30__interestingresults4.pdf')

















#
#    #%%
#    eps = 1e-3
#    remove_list = []
#    for node in exp.nodes():
#        inds = np.where(basis == node)
#        exp.nodes[node]['info'].hessian_sum = np.sum(np.diag(H0)[inds])
#        if exp.nodes[node]['info'].hessian_sum < eps:
#            print('node {} is below threshold'.format(node))
#            remove_list.append(node)
#    at_remove = {}
#    for node in list(exp.nodes()):
#        if node in remove_list:
#            exp.remove_component(node)
#        else:
#            at_remove[node] = at[node]
#    
#    exp.draw()
#
