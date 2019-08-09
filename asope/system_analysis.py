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

from assets.fitness_analysis import analysis_udr

plt.close("all")



env = OpticalField_Pulse(n_samples=2 ** 14, profile='gaussian', pulse_width=100e-12, f_rep=100e6, n_pulses=15, peak_power=1)
    
env.init_fitness(p=1, q=2)

# %%
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

#    exp.draw(node_label='disp_name')

exp.setattributes(at)
exp.simulate(env)
At = exp.nodes[exp.measurement_nodes[0]]['output']
fit = env.fitness(At)
print("Fitness: {}".format(fit))

#plt.figure()
#plt.plot(env.t, P(At))
#plt.plot(env.t, P(env.At0))
#plt.show()

print('Starting analysis')

#exp.init_fitness_analysis(at, env, method='MC', verbose=True)
#parameter_stability, others = exp.run_analysis(at, verbose=True)
#print(parameter_stability, others)
#
#exp.init_fitness_analysis(at, env, method='LHA', verbose=True)
#parameter_stability, others = exp.run_analysis(at, verbose=True)
#print(parameter_stability)

analysis_udr(at, exp, env, verbose=True)
