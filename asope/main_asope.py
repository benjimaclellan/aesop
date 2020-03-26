#%% import libraries
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import time
import matplotlib.pyplot as plt
import multiprocess as mp
from scipy import signal
import numpy as np
import copy

#%% import custom modules
from assets.functions import extractlogbook, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.waveforms import random_bit_pattern
from assets.graph_manipulation import change_experiment_wrapper, brand_new_experiment, remake_experiment
from assets.callbacks import save_experiment_and_plot

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from classes.experiment import Experiment

from config.config import config_parameters, config_topology

from optimization.ga_topology import geneticalgorithm_topology

from optimization.wrappers import optimize_experiment

plt.close("all")


#%%
if __name__ == '__main__': 

    # %% import all our configuration/hyperparameters for the optimization run
    CONFIG_TOPOLOGY = config_topology()
    CONFIG_PARAMETERS = config_parameters()
    
    # %%
    env = OpticalField_CW(n_samples=2 ** 14, window_t=10e-9, peak_power=1, lambda0=1.55e-6)
    target_harmonic = 12e9
    env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=False)

    #%%
    tstart = time.time()  
    hof, population, logbook = geneticalgorithm_topology(env, CONFIG_TOPOLOGY, CONFIG_PARAMETERS)
    tstop = time.time() 
    print('Total computation time: {}'.format(tstop-tstart))
    
    #%%
    for k in range(0,1):
        exp = remake_experiment(copy.deepcopy(hof[k]))
        exp.setattributes(hof[k].inner_attributes)
        exp.draw(node_label = 'both')
        exp.inject_optical_field(env.field)
            
        print(hof[k].inner_attributes)
        
        exp.simulate(env)
        
        field = exp.measure(env, exp.measurement_nodes[0], check_power=True)
#        plt.show()
        if field.ndim == 1:
            field = field.reshape(env.n_samples, 1)
        
        plt.figure()
        generated = env.shift_function(P(field))
        minval, maxval = np.min(generated), np.max(generated)
        if env.normalize:
            generated = (generated-minval)/(maxval-minval)
    
        plt.plot(env.t, generated,label='current')
        plt.plot(env.t, env.target,label='target',ls=':')
        plt.xlim([0,10/env.target_harmonic])
        plt.legend()
        plt.show()

        #%%
        # save_experiment_and_plot(exp, env, field)
    
