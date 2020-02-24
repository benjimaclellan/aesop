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
from scipy import signal
import seaborn

#%% import custom modules
from assets.functions import extractlogbook, save_class, load_class, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

from assets.waveforms import random_bit_pattern, bit_pattern, rf_chirp
from assets.callbacks import save_experiment_and_plot

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from classes.experiment import Experiment
from classes.geneticalgorithmparameters import GeneticAlgorithmParameters

from optimization.wrappers import optimize_experiment

plt.close("all")

#%%
if __name__ == '__main__':

    #%% store all our hyper-parameters for the genetic algorithm
    gap = GeneticAlgorithmParameters()
    gap.TYPE = "inner"
    gap.NFITNESS = 1            # how many values to optimize
    gap.WEIGHTS = (1.0,)    # weights to put on the multiple fitness values
    gap.MULTIPROC = False        # multiprocess or not
    gap.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gap.N_POPULATION = 80       # number of individuals in a population (make this a multiple of NCORES!)
    gap.N_GEN = 50               # number of generations
    gap.MUT_PRB = 0.01           # independent probability of mutation
    gap.CRX_PRB = 0.6           # independent probability of cross-over
    gap.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gap.VERBOSE = True          # verbose print statement for GA statistics
    gap.INIT = None
    gap.GRADIENT_DESCENT = 'numerical'
    gap.FINE_TUNE = True
    gap.ALPHA = 0.00005
    gap.MAX_STEPS = 2000
    gap.NUM_ELITE = 1
    gap.NUM_MATE_POOL = gap.N_POPULATION//2 - gap.NUM_ELITE

    #%% initialize our input pulse, with the fitness function too
    env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)
    target_harmonic = 12e9
    env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=False)

    components = {
        0: PhaseModulator(),
        1: WaveShaper()
    }
    adj = [(0,1)]


    #%% initialize the experiment, and perform all the pre-processing steps
    exp = Experiment()
    exp.buildexperiment(components, adj)
    exp.checkexperiment()

    exp.make_path()
    exp.check_path()
    exp.inject_optical_field(env.field)

    exp.draw(node_label = 'disp_name')
    plt.show()

    #%%
    exp, hof, hof_fine, log = optimize_experiment(exp, env, gap, verbose=True)
    at = copy.deepcopy(hof_fine[0])

    #%%
    exp.setattributes(at)
    exp.simulate(env)
    field = exp.nodes[exp.measurement_nodes[0]]['output']
    fit = env.fitness(field)
    print("Fitness: {}".format(fit))

    #%%
    plt.figure()
    plt.plot(env.t[0:1000], P(field[0:1000]))
    plt.plot(env.t[0:1000], P(env.field0[0:1000]))
    plt.show()

    #%%
    print('Starting analysis')

    exp.init_fitness_analysis(at, env, method='LHA', verbose=True)
    lha_stability, others = exp.run_analysis(at, verbose=True)

    # experiment.init_fitness_analysis(at, env, method='UDR', verbose=True)
    # udr_stability, others = experiment.run_analysis(at, verbose=True)
    #
    # experiment.init_fitness_analysis(at, env, method='MC', verbose=True)
    # mc_stability, others = experiment.run_analysis(at, verbose=True)

    xvals = np.arange(len(lha_stability))

    #%%
    plt.figure(figsize=[9,9])
    plt.stem(xvals - 0.1, lha_stability / np.max(lha_stability), label='LHA', markerfmt='go', linefmt='g--')
    # plt.stem(xvals, udr_stability/np.max(udr_stability), label='UDR', markerfmt = 'bo', linefmt = 'r--')
    # plt.stem(xvals+0.1, mc_stability/np.max(mc_stability), label='MC', markerfmt = 'ro', linefmt = 'b--')

    plt.legend()
    plt.show()

