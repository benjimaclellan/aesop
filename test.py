import time
import matplotlib.pyplot as plt
import multiprocess as mp
import uuid
import numpy as np
import networkx as nx
from sys import getsizeof

from assets.functions import extractlogbook, plot_individual, save_experiment, load_experiment, splitindices
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, BeamSplitter
from assets.classes import Experiment, GeneticAlgorithmParameters
from copy import copy, deepcopy
from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

## -----------------------------------
env = PulseEnvironment(p = 2, q = 1)
    
#a = (env.At) / (2) * np.exp(1j * np.pi/4)
#b = (env.At) / (2)
#
#c = a + b
#d = env.P(c)
#plt.plot(env.t, env.P(env.At),label='original')
#plt.plot(env.t, d, label='added')
#plt.legend()
#plt.show()

#components = [WaveShaper(), BeamSplitter(), PhaseModulator(), Fiber(), BeamSplitter(), Fiber()]
#adj = [ (0,1), (1,2), (1,3), (2,4), (3,4), (4,5) ]

components = [AWG(),Fiber(), PhaseModulator()]
adj = [ (0,1), (1,2) ]
nodes = [i for i in range(0,len(components))]


E = Experiment()
ind = [[3, 0.5603999676299348, 3.1417854328350674, 0.560363935929568], [3181.388628273651], [2.6622027565042767, 178999766.34779096, 2.400426236436889]]

E.add_nodes_from(nodes)
E.add_edges_from(adj)

E.n_components = E.number_of_nodes()
for i in range(E.number_of_nodes()):
    E.nodes[i]['title'] = components[i].name
    E.nodes[i]['info'] = components[i]
E.setattributes(ind)


def checkexperiment(E):
    ## check experiment
    mat = nx.adjacency_matrix(E).todense()
    isuptri = np.allclose(mat, np.triu(mat)) # check if upper triangular
    assert isuptri
    for i in range(E.number_of_nodes()):
        if E.in_degree()[i] == 0 :
            assert i == 0
        elif E.in_degree()[i] > 1:
            assert E.nodes[i]['info'].type == 'beamsplitter'
            assert E.out_degree()[i] == 1
        elif E.in_degree()[i] == 1:
            pass
        else:
            raise ValueError()
            
        if E.out_degree()[i] == 0:
            pass
        elif E.out_degree()[i] > 1:
            assert E.nodes[i]['info'].type == 'beamsplitter'
            assert E.in_degree()[i] == 1
        elif E.out_degree()[i] == 1:
            pass
        else:
            raise ValueError()
    return 

checkexperiment(E)

#def simexperiment(E, env):

E.nodes[0]['info'].simulate(env)
E[0][1]['env'] = env

for i in range(1,E.number_of_nodes()):
    pre = list(E.predecessors(i))
    suc = list(E.successors(i))
    
    for j in E.predecessors(i):
        E[j][i]['a'] = 'Edge-{}-{}'.format(i,j)
        env = E[j][i]['env']
 
    E.nodes[i]['info'].simulate(env)
    
    for s in E.successors(i):
        E[i][s]['env'] = env

print(getsizeof(E))
plot_individual(env, 0)
save_experiment('memorytest', E, env)

#simexperiment(E, env)
#nx.draw(E)
#plt.show()

