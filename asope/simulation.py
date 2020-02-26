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
#import multiprocess as mp

import copy
import autograd.numpy as np
import scipy.io as sio

#%% import custom modules
from assets.functions import extractlogbook, save_class, load_class, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

from assets.waveforms import random_bit_pattern, bit_pattern, rf_chirp
from assets.callbacks import save_experiment_and_plot

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse, OpticalField_PPLN
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, DelayLine
from classes.experiment import Experiment
#from classes.geneticalgorithmparameters import GeneticAlgorithmParameters

from config.config import config

from optimization.wrappers import optimize_experiment

plt.close("all")


#%%
if __name__ == '__main__':

    #%% initialize our input pulse, with the fitness function too
    env = OpticalField_PPLN(n_samples=2**16, window_t=1e-9, lambda0=1.55e-6, bandwidth=[1.54e-6, 1.56e-6])

    nodes = {0: DelayLine()}
    adj = []

    # %% initialize the experiment, and perform all the pre-processing steps
    exp = Experiment()
    exp.buildexperiment(nodes, adj)

    exp.checkexperiment()

    exp.make_path()
    exp.check_path()

    # at = {0: [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]}
    exp.
    exp.setattributes(at)

    exp.inject_optical_field(env.field)

    exp.simulate(env)
    field = exp.nodes[exp.measurement_nodes[0]]['output']

    fig, ax = plt.subplots(2,1)
    ax[0].plot(env.t/1e-12, P(field), label='Field Out', alpha=0.5)
    ax[0].plot(env.t/1e-12, P(env.field), label='Field In', alpha=0.5)
    ax[0].set(xlabel='Time (ps)', ylabel='AU')

    ax[1].plot(env.c0/(env.f + env.f0)/1e-6, P(FFT(field, env.dt)), label='Field Out', alpha=0.5)
    ax[1].plot(env.c0/(env.f + env.f0)/1e-6, P(FFT(env.field, env.dt)), label='Field In', alpha=0.5)
    ax[1].set(xlabel=r'Wavelength ($\mu$m)', ylabel='AU')
    for axi in ax:
        axi.legend()
    plt.show()
