
import time
import matplotlib.pyplot as plt
import multiprocess as mp
import uuid
import numpy as np
import networkx as nx

from assets.functions import extractlogbook, plot_individual, save_experiment, load_experiment, splitindices
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, BeamSplitter, FrequencySplitter
from assets.classes import Experiment, GeneticAlgorithmParameters
from copy import copy, deepcopy
from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

## -----------------------------------
env = PulseEnvironment(p = 2, q = 1)
    
#components = [FrequencySplitter(), AWG(), Fiber(), AWG(), Fiber()]#, BeamSplitter()]
#adj = [ (0,1), (1,2), (0,3), (3,4)]#, (2,5), (4,5) ]
##ind = [[0.5], [2, np.pi/2, 3*np.pi/2] , [1], [2, 0, np.pi], [4], [0.5]]
#ind = [ comp.newattribute() for comp in components ]
#nodes = [i for i in range(0,len(components))]


components = [FrequencySplitter(), Fiber(), Fiber(), BeamSplitter()]
adj = [ (0,1), (0,2), (2,3), (1,3) ]
ind = [[0.00], [10], [0.5], [0.5] ]
#ind = [ comp.newattribute() for comp in components ]
nodes = [i for i in range(0,len(components))]


E = Experiment()


E.add_nodes_from(nodes)
E.add_edges_from(adj)
E.terminal_nodes = []

E.n_components = E.number_of_nodes()
for i in range(E.number_of_nodes()):
    E.nodes[i]['title'] = components[i].name
    E.nodes[i]['info'] = components[i]
E.setattributes(ind)


E.checkexperiment()
E.simulate(env)


plt.show()
    
    
plt.figure()
E.draw()

for terminal_node in E.terminal_nodes:
    for p in E.predecessors(terminal_node):
        env = E[p][terminal_node]['env']
        E.plot_env(env)
plt.show()

