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

#%% import custom modules
from assets.functions import extractlogbook, save_experiment, load_experiment, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.waveforms import random_bit_pattern
from assets.graph_manipulation import change_experiment_wrapper, brand_new_experiment, remake_experiment

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, AmplitudeModulator
from classes.experiment import Experiment
from classes.geneticalgorithmparameters import GeneticAlgorithmParameters

from optimization.geneticalgorithmouter import outer_geneticalgorithm

from asope_inner import optimize_experiment

#import warnings
#warnings.filterwarnings("ignore")

plt.close("all")


#%%
if __name__ == '__main__': 
    
    #%% store all our hyper-parameters for the genetic algorithm
    gapO = GeneticAlgorithmParameters()
    gapO.TYPE = "outer"
    gapO.NFITNESS = 1            # how many values to optimize
    gapO.WEIGHTS = (1.0),    # weights to put on the multiple fitness values
    gapO.MULTIPROC = False       # multiprocess or not
    gapO.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gapO.N_POPULATION = 5       # number of individuals in a population
    gapO.N_GEN = 3              # number of generations
    gapO.MUT_PRB = 1.0           # independent probability of mutation
    gapO.CRX_PRB = 0.0           # independent probability of cross-over
    gapO.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gapO.VERBOSE = 1             # verbose print statement for GA statistics
    gapO.INIT = None
    gapO.NUM_ELITE = 1
    gapO.NUM_MATE_POOL = gapO.N_POPULATION - gapO.NUM_ELITE
#    gapO.POTENTIAL_COMPS = {'splitters':[PowerSplitter, FrequencySplitter], 'nonsplitters':[Fiber, PhaseModulator, WaveShaper, AmplitudeModulator]}
    gapO.POTENTIAL_COMPS = {'splitters':[], 'nonsplitters':[PhaseModulator, WaveShaper]}
    
    #%%
    gapI = GeneticAlgorithmParameters()
    gapI.TYPE = "inner"
    gapI.NFITNESS = 1           # how many values to optimize
    gapI.WEIGHTS = (1.0),     # weights to put on the multiple fitness values
    gapI.MULTIPROC = True        # multiprocess or not
    gapI.NCORES = mp.cpu_count() # number of cores to run multiprocessing with
    gapI.N_POPULATION = 20      # number of individuals in a population
    gapI.N_GEN = 20              # number of generations
    gapI.MUT_PRB = 0.5           # independent probability of mutation
    gapI.CRX_PRB = 0.5          # independent probability of cross-over
    gapI.N_HOF = 1               # number of inds in Hall of Fame (num to keep)
    gapI.VERBOSE = 0             # verbose print statement for GA statistics
    gapI.INIT = None
    gapI.FINE_TUNE = False
    gapI.NUM_ELITE = 1
    gapI.NUM_MATE_POOL = gapI.N_POPULATION//2 - gapI.NUM_ELITE
    
    
    #%%
    env = OpticalField_CW(n_samples=2**12, window_t=10e-9, peak_power=1)    
    target_harmonic = 12e9
    env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=True)

    #%%
    tstart = time.time()  
    hof, population, logbook = outer_geneticalgorithm(env, gapO, gapI)
    tstop = time.time() 
    
    
    #%%
    for k in range(0,1):
        exp = remake_experiment(hof[k])
        exp.setattributes(hof[k].inner_attributes)
        exp.draw(node_label = 'both')
        exp.inject_optical_field(env.At)
            
        print(hof[k].inner_attributes)
        
        exp.simulate(env)
        
        At = exp.measure(env, exp.measurement_nodes[0], check_power=True)
        plt.show()
        
        env.compare(At)
        
        exp.visualize()
        
        save_experiment("results/current", exp)
    
