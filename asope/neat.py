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
from classes.geneticalgorithmparameters import GeneticAlgorithmParameters

from optimization.geneticalgorithmouter import outer_geneticalgorithm

from asope_inner import optimize_experiment

components = {
    0:PowerSplitter(),
	1:Fiber(),
	2:Fiber(),
	3:PowerSplitter()
    }
adj = [(0,1), (1,3), (0,2), (2,3)]


#%% initialize the experiment, and perform all the pre-processing steps
exp = Experiment()
exp.buildexperiment(components, adj)
exp.checkexperiment()

exp.make_path()
exp.check_path()

exp.draw(node_label = 'disp_name')
plt.show()
