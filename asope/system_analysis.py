import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings("ignore")

# %% import public modules
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocess as mp
import copy
import autograd.numpy as np
import pandas as pd
from scipy import signal
import seaborn

# %% import custom modules
#from assets.functions import extractlogbook, save_class, load_class, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter,  AmplitudeModulator
from classes.experiment import Experiment


plt.close("all")

#env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)
#target_harmonic = 12e9
#env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=False)

env = OpticalField_Pulse(n_samples=2 ** 14, profile='gaussian', pulse_width=100e-12, f_rep=100e6, n_pulses=15, peak_power=1)
env.init_fitness(p=1, q=2)

# %%
#components = {
#        
#    0: WaveShaper()
#}
#adj = []

#components = {
#                0:PhaseModulator(),
#                1:WaveShaper(),
#             }
#adj = [(0,1)]
#at = {   0: [ 0.8234132112176106, 
#              12000000000.0],
#         1: [ 0.6060667463824007,
#              0.0,
#              0.721833819341576,
#              0.9503376429345127,
#              0.6234058812515285,
#              4.888443104871878,
#              0.4101923968206471,
#              1.9730094674171068,
#              0.5122717909947406,
#              2.191008207394641]}

components = {
    0: Fiber(),
    1: Fiber(),
    2: PhaseModulator()
}
adj = [(0, 1), (1, 2)]

at = { 0: [2922.346465662926],
       1: [61.44445153456259],
       2: [0.12755565759819779, 21000000000.0]}

# %% initialize the experiment, and perform all the pre-processing steps
exp = Experiment()
exp.buildexperiment(components, adj)
exp.checkexperiment()

exp.make_path()
exp.check_path()
exp.inject_optical_field(env.At)

at = exp.newattributes()

exp.setattributes(at)
exp.simulate(env)
At = exp.nodes[exp.measurement_nodes[0]]['output']
fit = env.fitness(At)
print("Fitness: {}".format(fit))

#plt.figure()
#plt.plot(env.t, P(At))
#plt.plot(env.t, P(env.At0))
#plt.show()

#%%
print('Starting analysis')

exp.init_fitness_analysis(at, env, method='LHA', verbose=True)
lha_stability, *others = exp.run_analysis(at, verbose=True)

exp.init_fitness_analysis(at, env, method='UDR', verbose=True)
udr_stability, *others = exp.run_analysis(at, verbose=True)

exp.init_fitness_analysis(at, env, method='MC', verbose=True)
mc_stability, *others = exp.run_analysis(at, verbose=True)

xvals = np.arange(len(udr_stability))

#%%


plt.figure(figsize=[9,9])
plt.stem(xvals-0.2, lha_stability/np.max(np.abs(lha_stability)), label='LHA', markerfmt = 'go', linefmt = 'g--')
plt.stem(xvals, udr_stability/np.max(udr_stability), label='UDR', markerfmt = 'ro', linefmt = 'r--')
plt.stem(xvals+0.2, mc_stability/np.max(mc_stability), label='MC', markerfmt = 'bo', linefmt = 'b--')

plt.legend()
plt.show()
