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
import multiprocess as mp

import copy 
import autograd.numpy as np
from scipy import signal

#%% import custom modules
from assets.functions import extractlogbook, save_class, load_class, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

from assets.waveforms import random_bit_pattern, bit_pattern, rf_chirp
from assets.callbacks import save_experiment_and_plot

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from classes.experiment import Experiment
#from classes.geneticalgorithmparameters import GeneticAlgorithmParameters

from config.config import config

from optimization.wrappers import optimize_experiment

plt.close("all")

#%%
if __name__ == '__main__':
    gap = config()

    #%% initialize our input pulse, with the fitness function too
    env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)
    target_harmonic = 12e9
    env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=False)

    # genotype = {'nodes': {0: PhaseModulator, 1: WaveShaper},
    #             'adj': [(0,1)]
    #             }
    nodes = {0: PhaseModulator, 1: WaveShaper}
    adj = [(0, 1)]

    #%% initialize the experiment, and perform all the pre-processing steps
    exp = Experiment()
    exp.buildexperiment(nodes, adj)
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

