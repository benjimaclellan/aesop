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
import scipy.signal as sig

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


def fsr_value(psd, peak_inds, peak_widths):
    fitness = np.mean(np.diff(peak_inds)) / (np.std(np.diff(peak_inds) + 1)) #* np.mean(psd)
    return fitness

def q_factor(psd, peak_inds, peak_widths):
    fitness = 1/np.mean(peak_widths)
    return fitness

def get_peaks(psd):
    # psd = PSD(field, env.dt, env.df)
    peak_array = np.squeeze(psd / max(psd))
    peak_inds, prop = sig.find_peaks(peak_array,
                                     height=None, threshold=None, distance=None, prominence=0.5, width=None,
                                     wlen=None,
                                     rel_height=None, plateau_size=None)
    (peak_widths, width_heights, *_) = sig.peak_widths(peak_array, peak_inds, rel_height=0.5,
                                                       prominence_data=None,
                                                       wlen=None)
    return peak_inds, peak_widths


#%%
if __name__ == '__main__':

    #%% initialize our input pulse, with the fitness function too
    # env = OpticalField_PPLN(n_samples=2**16, window_t=2.0e-9, lambda0=1.55e-6, bandwidth=[1.53e-6, 1.57e-6])
    env = OpticalField_Pulse(n_samples=2**15, profile='sech', pulse_width=1.0e-12, train=True, t_rep=100e-12, window_t=2.0e-9, peak_power=1, lambda0=1.55e-6)

    nodes = {0: DelayLine()}
    adj = []

    # %% initialize the experiment, and perform all the pre-processing steps
    exp = Experiment()
    exp.buildexperiment(nodes, adj)
    exp.make_path()

    # at = {0: [0.5, 0.5, 0.5, 0.5]}
    fig, ax = plt.subplots(2, 1, figsize=[10,10])
    for i in range(1):
        for axi in ax:
            axi.cla()

        at = exp.newattributes()
        exp.setattributes(at)
        exp.inject_optical_field(env.field0)
        exp.simulate(env)
        field = exp.nodes[exp.measurement_nodes[0]]['output']

        p0 = P(env.field0)
        p = P(field) /max(p0)
        p0 *= 1 / max(p0)

        psd0 = PSD(env.field0, env.dt, env.df)
        psd = PSD(field, env.dt, env.df) /max(psd0)
        psd0 *= 1 / max(psd0)

        ax[0].plot(env.t/1e-12, p, label='Field Out', alpha=0.5)
        ax[0].plot(env.t/1e-12, p0, label='Field In', alpha=0.5)
        ax[0].set(xlabel='Time (ps)', ylabel='AU')
        # ax[0].set(xlim=[-200, 200])

        ax[1].plot(env.c0/(env.f + env.f0)/1e-6, psd, label='Field Out', alpha=0.5)
        ax[1].plot(env.c0/(env.f + env.f0)/1e-6, psd0, label='Field In', alpha=0.5)

        ax[1].set(xlabel=r'Wavelength ($\mu$m)', ylabel='AU')
        # ax[1].set(xlim=[1.52, 1.58])
        for axi in ax:
            axi.legend()

        print(at)
        # plt.pause(0.05)
        # plt.waitforbuttonpress()
