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
import random

#%% import custom modules
from assets.functions import extractlogbook, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.waveforms import random_bit_pattern
from assets.graph_manipulation import change_experiment_wrapper, brand_new_experiment, remake_experiment
from assets.callbacks import save_experiment_and_plot

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from classes.components import Input, Output
from classes.experiment import Experiment
from classes.geneticalgorithmparameters import GeneticAlgorithmParameters

from optimization.geneticalgorithmouter import outer_geneticalgorithm

from asope_inner import optimize_experiment

env = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)
target_harmonic = 12e9
env.init_fitness(0.5*(signal.sawtooth(2*np.pi*target_harmonic*env.t, 0.5)+1), target_harmonic, normalize=False)

components = {
	0: Input(),
	1: Output(),
	}
adj = [(0, 1)]


#%% initialize the experiment, and perform all the pre-processing steps
exp = Experiment()
exp.buildexperiment(components, adj)
#exp.checkexperiment()

exp.make_path()
exp.check_path()
exp.inject_optical_field(env.field)

exp.draw(node_label='disp_name')
plt.show()

def newgene(history, edge):
	return {'HISTORY': history, 'FROM': edge[0], 'TO': edge[1], 'EDGE': edge, 'STATUS': 'ENABLED'}

exp.history = [newgene(0, (0, 1))]
# options = [Fiber, PhaseModulator, WaveShaper, PowerSplitter]
options = [Fiber, PowerSplitter]

def changegraph(experiment):
	print('\n')
	choice = random.choice(options)
	number = experiment.number_of_nodes()

	edge = random.choice(list(filter(lambda d: d['STATUS'] == "ENABLED", exp.history)))["EDGE"]
	i = next(i for i, item in enumerate(experiment.history) if item['EDGE'] == edge)
	if choice == PowerSplitter:
		print('Add a interferometer')
		experiment.add_node(number, info=choice())
		experiment.add_node(number+1, info=choice())

		experiment.add_edge(edge[0], number)
		experiment.add_edge(number, number+1)
		experiment.add_edge(number, number+1)
		experiment.add_edge(number+1, edge[1])

		experiment.history.append(newgene(len(experiment.history), (edge[0], number)))
		experiment.history.append(newgene(len(experiment.history), (number, number+1)))
		experiment.history.append(newgene(len(experiment.history), (number, number+1)))
		experiment.history.append(newgene(len(experiment.history), (number+1, edge[1])))


	else:
		experiment.add_node(number, info=choice())

		experiment.add_edge(edge[0], number)
		experiment.add_edge(number, edge[1])

		experiment.history.append(newgene(len(experiment.history), (edge[0], number)))
		experiment.history.append( newgene(len(experiment.history), (number, edge[1])))


	print(i)
	experiment.history[i]['STATUS'] = 'DISABLED'
	experiment.remove_edge(edge[0], edge[1])


	return

for j in range(0,10):
	changegraph(exp)

for item in exp.history:
	print(item)

exp.draw(node_label='disp_name')
plt.show()

exp.make_path()
exp.check_path()
# exp.inject_optical_field(env.field)
#
# at = exp.newattributes()
# exp.setattributes(at)
# exp.simulate(env)
# field = exp.nodes[exp.measurement_nodes[0]]['output']
# fit = env.fitness(field)
# print("Fitness: {}".format(fit))
#
# plt.figure()
# plt.plot(env.t, P(field))
# plt.plot(env.t, P(env.field0))
# plt.show()