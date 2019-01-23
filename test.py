import time
import matplotlib.pyplot as plt
import multiprocess as mp
import uuid
import numpy as np
import networkx as nx

from assets.functions import extractlogbook, plot_individual, save_experiment, load_experiment, splitindices
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper
from assets.classes import Experiment, GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

## -----------------------------------
env = PulseEnvironment(p = 2, q = 1)
    


#    components = [AWG(), Fiber()]
components = [WaveShaper(), PhaseModulator(), Fiber()]
experiment = Experiment()
experiment.buildexperiment(components)

env.reset()
individual = experiment.newattributes()
experiment.setattributes(individual)
experiment.simulate(env)
fitness = env.fitness()
#plot_individual(env, fitness)
#experiment.visualize(env)
#plt.show()

adj = nx.adjacency_matrix(experiment)
print(adj.todense())
