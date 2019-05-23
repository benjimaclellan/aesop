#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

#%% import custom modules
from assets.functions import extractlogbook, save_experiment, load_experiment, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.waveforms import random_bit_pattern, rf_chirp
from assets.graph_manipulation import change_experiment_wrapper
from assets.callbacks import save_experiment_and_plot

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, AmplitudeModulator
from classes.experiment import Experiment

plt.close("all")

#%%

env = OpticalField_CW(n_samples=2**14+1, window_t=10e-9, peak_power=1)

target_harmonic = 24e9
#env.init_fitness(0.5*(signal.square(2*np.pi*target_harmonic*env.t)+1), target_harmonic, normalize=False)


target_harmonic = 24e9
target = rf_chirp(env.t, target_harmonic, 4/target_harmonic, 1.1*target_harmonic)
plt.figure()
plt.plot(env.t, target)
