
import time
import matplotlib.pyplot as plt
import multiprocess as mp
import uuid
import numpy as np
import networkx as nx

from assets.functions import extractlogbook, plot_individual, save_experiment, load_experiment, splitindices
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from assets.classes import Experiment, GeneticAlgorithmParameters
from copy import copy, deepcopy
from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

## -----------------------------------
env = PulseEnvironment(p = 3, q = 2)
    
components = [FrequencySplitter(), AWG(), Fiber(), AWG(), Fiber(), PowerSplitter()]
adj = [ (0,1), (0,3), (1,2), (3,4), (4,5), (2,5)]
terminal = 5

ind = [[0.02], [1, -1.6076094355304262], [1592.0], [1, -1.6355283141702814], [1311.0], [0.5]]

E = Experiment()
E.buildexperiment(components, adj)

E.terminal = 'terminal{}'.format(terminal)

E.setattributes(ind)
E.checkexperiment()

plt.figure()
E.draw()


env = E.simulate(env)
E.plot_env(env)
fitness = env.fitness()
print(fitness)
