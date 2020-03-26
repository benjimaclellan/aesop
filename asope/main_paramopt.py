"""
main_paramopt.py
Benjamin MacLellan 2020

Optimizes the control parameters of a given optical system (no topology optimization)
"""

#%% this allows proper multiprocessing (overrides internal multiprocessing settings)
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import warnings
warnings.filterwarnings("ignore")

# %% import public modules
import matplotlib.pyplot as plt
import copy 
import autograd.numpy as np
from scipy import signal

# %% import custom modules
from assets.functions import extractlogbook, save_class, load_class, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

from assets.waveforms import random_bit_pattern, bit_pattern, rf_chirp
from assets.callbacks import save_experiment_and_plot

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, DelayLine
from classes.experiment import Experiment

from optimization.wrappers import optimize_experiment

from config.config import import_config

plt.close("all")

#%%
if __name__ == '__main__':

    #%% import the config settings for this optimization run
    CONFIG_PARAMETERS = import_config('config.config_parameters_2020')

    #%% initialize our input pulse, with the fitness function too
    env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1, lambda0=1.55e-6)
    target_harmonic = 12e9
    env.init_fitness(0.5*(signal.square(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=False)

    #%% setup the system to consider
    # nodes = {0: PhaseModulator(), 1: WaveShaper()}
    # adj = [(0, 1)]

    nodes = {0: PowerSplitter(),
             1: PhaseModulator(),
             2: PowerSplitter()}
    adj = [(0, 1), (1, 2), (0, 2)]

    #%% initialize the experiment, and perform all the pre-optimization steps (checking for issues)
    exp = Experiment()
    exp.buildexperiment(nodes, adj)
    exp.checkexperiment()

    exp.make_path()
    exp.check_path()
    exp.inject_optical_field(env.field)

    #%% run the optimization based on given configurations
    exp, hof, hof_fine, log = optimize_experiment(exp, env, CONFIG_PARAMETERS, verbose=True)

    # take the best-performing individual of all runs as the optimal settings
    at = copy.deepcopy(hof_fine[0])

    # %% show the system topology
    exp.draw(node_label='disp_name')
    plt.show()

    #%% simulate the system with the best settings to visualize its performance
    exp.setattributes(at)
    exp.simulate(env)
    field = exp.nodes[exp.measurement_nodes[0]]['output']
    fit = env.fitness(field)
    print("Fitness: {}".format(fit))

    plt.figure()
    plt.plot(env.t[0:1000], P(field[0:1000]))
    plt.plot(env.t[0:1000], P(env.field0[0:1000]))
    plt.show()

    # #%% check hessian analysis
    # print('Starting analysis')
    #
    # exp.init_fitness_analysis(at, env, method='LHA', verbose=True)
    # lha_stability, others = exp.run_analysis(at, verbose=True)
    # xvals = np.arange(len(lha_stability))
    #
    # plt.figure(figsize=[9,9])
    # plt.stem(xvals - 0.1, lha_stability / np.max(lha_stability), label='LHA', markerfmt='go', linefmt='g--')
    #
    # plt.legend()
    # plt.show()

