"""
Copyright Benjamin MacLellan

Testing for ASOPE. Simulates a single experiment.

"""
#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

#%% import custom modules
from assets.functions import extractlogbook, save_experiment, load_experiment, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.waveforms import random_bit_pattern
from assets.graph_manipulation import change_experiment_wrapper

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, AmplitudeModulator
from classes.experiment import Experiment

plt.close("all")

import warnings
warnings.filterwarnings("ignore")

#%% define the optical field. here we use a CW source, all units are SI
env = OpticalField_CW(n_samples=2**13+1, window_t=10e-9, peak_power=1)

target_harmonic = 12e9
#env.init_fitness(0.5*(signal.square(2*np.pi*target_harmonic*env.t)+1), target_harmonic, normalize=False)

target, bit_sequence = random_bit_pattern(env.n_samples, 8, target_harmonic, None, env.dt)
env.init_fitness(target, target_harmonic, normalize=False)
env.bit_sequence = bit_sequence

#%% define experimental setup
components = {
                0:PhaseModulator(),
                1:WaveShaper()
             }
adj = [(0,1)]

exp = Experiment()
exp.buildexperiment(components, adj)
exp.cleanexperiment()
exp.checkexperiment()
exp.make_path()
exp.check_path()

exp.inject_optical_field(env.At)

exp.draw(node_label='both')

#%% make a random set of parameters (attributes) and simulate
at = exp.newattributes()

exp.setattributes(at)
exp.simulate(env)
exp.visualize(env, exp.measurement_nodes[0])
plt.show()

At = exp.nodes[exp.measurement_nodes[0]]['output']

exp.measure(env, exp.measurement_nodes[0], check_power=True)
fit = env.fitness(At)
print(fit)

print(env.bit_sequence)
plt.figure()
plt.plot(env.t, env.target)

#%% Here we can see lots of graph configurations
#fig = plt.figure()
#for i in range(0,100):
#    print("Let's change the graph - iteration ",i)
#    exp, _ = change_experiment_wrapper(exp, {'splitters':[PowerSplitter, FrequencySplitter], 'nonsplitters':[Fiber, PhaseModulator, WaveShaper, AmplitudeModulator]})
#    plt.clf()
#    exp.draw(node_label='both', title='Something', fig=fig)
#    plt.pause(0.1)