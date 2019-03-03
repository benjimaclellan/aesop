
import time
import matplotlib.pyplot as plt
import multiprocess as mp
import uuid
import numpy as np
import networkx as nx

from assets.functions import extractlogbook, plot_individual, save_experiment, load_experiment, splitindices
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, Detector
from assets.classes import Experiment, GeneticAlgorithmParameters
from copy import copy, deepcopy
from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

## -----------------------------------
env = PulseEnvironment(p = 3, q = 2)
    
measurement_nodes = []

#components = (
#    {
#     -1: Fiber(),
#     1:AWG(),
#     2:Fiber()
#    })
#adj = [(-1,1), (1,2)]
#measurement_nodes = [2]
#attributes = (
#    {
#     -1:[0],
#     1:[1,0],
#     2:[2]
#    })


#components = (
#    {
#     -1: Fiber(),
#     0:PowerSplitter(),
#     1:Fiber(),
#     2:Fiber(),
#     3:PowerSplitter(),
#    }
#    )
#adj = [(-1,0), (0,1), (0,2), (1,3), (2,3)]
#measurement_nodes = [3]
#attributes = {
#                -1:[0],
#                1:[0],
#                2:[2]
#             }



components = (
    {
     -2:Fiber(),
     -1:Fiber(),
     0:PowerSplitter(),
     1:Fiber(),
     2:Fiber(),
     3:Fiber(),
     4:Fiber(),
     5:PowerSplitter(),
    }
    )
adj = [ (-2,-1), (-1,0), (0,1), (0,3), (1,2), (3,4), (4,5), (2,5) ]
measurement_nodes = [5]
attributes = (
    {
     -2:[0],
     -1:[0],
     1:[1],
     2:[0],
     3:[0],
     4:[0]
    })


#components = (
#    {
#     'q':PhaseModulator(),
#     's':AWG(),
#     'd':AWG(),
#     'f':AWG(),
#     -2:FrequencySplitter(),
#     -1:Fiber(),
#     0:AWG(),
#     'tt': AWG(),
#     1:FrequencySplitter(),
##     2:FrequencySplitter(),
#     3:AWG(),
#     'dd':AWG(),
#     4:Fiber(),
#     'a':AWG(),
#     6:FrequencySplitter(),
#     7:Detector(),
#     'benny':Detector()
#    }
#    )

#adj = [ (1,'dd'), ('q',6), ('s','d'), ('d', 'f'), ('f', -2), (-2,-1), (-2,0), (0,1), (-1,1), (1,3), (1,4), (3,'a'), (4,6), ('a',6), (-2, 'tt'), ('tt', 6), (6,7), (7, 'benny')]


E = Experiment()
E.buildexperiment(components, adj)
E.checkexperiment()
E.measurement_nodes = measurement_nodes

plt.figure()
E.draw(titles = 'both')
plt.show()
plt.pause(0.1)

E.make_path()
E.check_path()

#print(E.path)

#attributes = E.newattributes()
E.setattributes(attributes)

At = env.At
for path_i, subpath in enumerate(E.path):
    
    for ii, node in enumerate(subpath):  
        
        if E.nodes[node]['info'].splitter:
#            print('this is a splitter', node)
            At = np.zeros([env.N, len(E.pre(node))]).astype('complex')
            for jj in range(len(E.pre(node))):
                At[:, jj] = E[E.pre(node)[jj]][node]['At']
            
#            print(node)
            
            At = E.nodes[node]['info'].simulate(env, At, max(1,len(E.suc(node))))
            
#            if len(E.suc(node)) == 0:
#                E.nodes[node]['output'] = At
#            else:
            for jj in range(len(E.suc(node))):
                E[node][E.suc(node)[jj]]['At'] = At[:,jj]
                
        else:
#            print('this is NOT a splitter', node)
            
            if ii == 0:
                if len(E.pre(node)) == 0:
                    At = env.At 
                else:
                    At = E[E.pre(node)[ii]][node]['At']
            
            At = E.nodes[node]['info'].simulate(env, At) 
            
            if ii == len(subpath)-1 and len(E.suc(node)) > 0: # last node in subpath (need to save now)
#                print('and were at the end')

                At = E.nodes[node]['info'].simulate(env, At)
                E[node][E.suc(node)[0]]['At'] = At
#                print(node, E.suc(node)[0])
        
        if node in E.measurement_nodes:
            E.nodes[node]['output'] = At
        

for measurement_node in E.measurement_nodes:
    
    At = E.nodes[measurement_node]['output'].reshape(env.N)
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), dpi=80)
    ax[0].set_title('Measurement node {}: {}'.format(measurement_node, E.nodes[measurement_node]['title']))
    alpha = 0.4
    ax[0].plot(env.t, env.P(env.At0), lw = 4, label='Input', alpha=alpha)
    ax[0].plot(env.t, env.P(At), ls='--', label='Output')    
    ax[0].legend()
    
    Af = env.FFT(At, env.dt)
    ax[1].plot(env.f, env.PSD(env.Af0, env.df), lw = 4, label='Input', alpha=alpha)
    ax[1].plot(env.f, env.PSD(Af, env.df), ls='-', label='Output')
#    print(max(env.PSD(env.Af0, env.df)))
#    print(max(env.PSD(Af[:,0], env.df)))
    ax[1].legend()
    

plt.show()

