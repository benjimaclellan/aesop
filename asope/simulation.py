"""
simulation.py
Benjamin MacLellan 2020

Script to test new components, or simply to simulate a topology + parameter set of interest
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

import copy
import autograd.numpy as np
import scipy.io as sio
import scipy.signal as sig

#%% import custom modules
from assets.functions import extractlogbook, save_class, load_class, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.functions import scale_units

from assets.waveforms import random_bit_pattern, bit_pattern, rf_chirp
from assets.callbacks import save_experiment_and_plot

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse, OpticalField_PPLN
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, DelayLine
from classes.experiment import Experiment

from optimization.gradientdescent import pack_opt, unpack_opt

from config.import_config import import_config

plt.close("all")

#%%
if __name__ == '__main__':

    #%% initialize our input optical field, with the objective function of interest
    # env = OpticalField_PPLN(n_samples=2**16, window_t=2.0e-9, lambda0=1.55e-6, bandwidth=[1.53e-6, 1.57e-6])
    env = OpticalField_Pulse(n_samples=2**16, window_t=500e-12, profile='gaussian', pulse_width=100e-15, train=True, t_rep=100e-12, peak_power=1, lambda0=1.55e-6)
    # env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1, lambda0=1.55e-6, normalize=False)

    config = import_config('./config/config_topology_2020')

    # %% set-up the system

    nodes = {0: DelayLine()}

    adj = []

    #%% define the attributes (control parameters)
    at = {0: 5*[0.5]}

    # %% initialize the experiment, and perform all the pre-processing steps
    exp = Experiment()
    exp.buildexperiment(nodes, adj)
    exp.make_path()

    # setup figure for visualizing
    fig, ax = plt.subplots(2, 1, figsize=[10,10])
    for i in range(1):
        for axi in ax:
            axi.cla()

        # at = exp.newattributes()
        # inject optical field, simulate system, and retrieve results
        exp.setattributes(at)
        exp.inject_optical_field(env.field0)
        exp.simulate(env)
        field = exp.nodes[exp.measurement_nodes[0]]['output']

        # calculate power in temporal & spectral domains & plots (normalized)
        p0 = P(env.field0)
        p = P(field) /max(p0)
        p0 *= 1 / max(p0)

        psd0 = PSD(env.field0, env.dt, env.df)
        psd = PSD(field, env.dt, env.df) /max(psd0)
        psd0 *= 1 / max(psd0)

        ax[0].plot(env.t, p, label='Field Out', alpha=0.5)
        ax[0].plot(env.t, p0, label='Field In', alpha=0.5)
        ax[0].set(xlabel='Time', ylabel='AU')
        scale_units(ax[0], unit='s', axes=['x'])

        ax[1].plot(env.c0/(env.f + env.f0), psd, label='Field Out', alpha=0.5)
        ax[1].plot(env.c0/(env.f + env.f0), psd0, label='Field In', alpha=0.5)

        ax[1].set(xlabel=r'Wavelength', ylabel='AU')
        scale_units(ax[1], unit='m', axes=['x'])

        for axi in ax:
            axi.legend()

        print(at)