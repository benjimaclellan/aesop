

#%%
import copy 
from random import shuffle, sample
import time
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import networkx as nx
import keyboard

from assets.functions import extractlogbook, save_experiment, load_experiment, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, AmplitudeModulator
from assets.classes import Experiment

from assets.graph_manipulation import *

plt.close("all")

import warnings
warnings.filterwarnings("ignore")

#%%
for i in range(0,1):
    plt.close('all')
    
    env = PulseEnvironment()
    
    
    components = {
    1:PhaseModulator(),
    0:WaveShaper(),
    2:Fiber()
    }
    adj = [(0,1),(1,2)]
    measurement_nodes=[2]
    E = Experiment()
    E.buildexperiment(components, adj, measurement_nodes)
    
#    E = Experiment()
#    E,_ = brand_new_experiment(E)
    
#    for i in range(5):
#        E, _ = change_experiment_wrapper(E)
    
    E.cleanexperiment()
    E.checkexperiment()
    
    E.make_path()
    E.check_path()
    
    E.draw(node_label='both')
    
    for node in E.nodes():
        if not E.pre(node):
            E.nodes[node]['input'] = env.At
    
    E.measurement_nodes = []
    for node in E.nodes():
        if not E.suc(node):
            E.measurement_nodes.append(node)
        
    at = E.newattributes()
    #at = {0:[.7,1e8,0.]}
    
    E.setattributes(at)
    E.simulate(env)
    E.visualize(env, E.measurement_nodes[0])
    
    E.measure(env, E.measurement_nodes[0], check_power=True)
    
    plt.show()
    plt.pause(1.5)
    


#%%
"""
env = PulseEnvironment(p = 2, q = 1, profile = 'cw')

E = Experiment()
E,_ = brand_new_experiment(E)

fig1 = plt.figure()
fig2 = plt.figure()

for i in range(60):
    for j in range(2):
        E, _ = change_experiment_wrapper(E)
    E.cleanexperiment()
    E.make_path()
    E.check_path()
    
    mapping=dict(zip( [item for sublist in E.path for item in sublist],range(0,len(E.nodes()))))
    E = nx.relabel_nodes(E, mapping) 
    
    E.make_path()
    E.check_path()
    E.checkexperiment()   

    plt.figure(fig1.number)
    plt.clf()
    E.draw(node_label='disp_name', title='Optical Setup Mutation')#, fig=fig1)
    
#    plt.savefig('results/{}_{}_graph.png'.format(i, env.profile), bbox='tight', dpi=300)
    plt.show()
    
    
    E.injection_nodes = []
    E.measurement_nodes = []
    for node in E.nodes():
        if len(E.pre(node)) == 0:
            E.injection_nodes.append(node)
            E.nodes[node]['input'] = env.At
        if len(E.suc(node)) == 0:
            E.measurement_nodes.append(node)
        
    E.make_path()
    print(E.path)
    E.check_path()
        
    
    
    at = E.newattributes()
    E.setattributes(at)
    
    plt.figure(fig2.number)
    E.simulate(env)
    power_check = E.measure(env, E.measurement_nodes[0], fig=fig2, check_power = True)  
    plt.show()
#    plt.savefig('results/{}_{}_output.png'.format(i, env.profile), bbox='tight', dpi=300)

    if not power_check:
        raise ValueError

    adj = list(E.edges())
    components = {}
    measurement_nodes = []
    for node in E.nodes:
#        components[node] = eval(E.nodes[node]['info'].__class__.__name__)()
        components[node] = E.nodes[node]['info'].__class__.__name__
        if len(E.suc(node)) == 0:
            measurement_nodes.append(node)
    print('components = {')
    for node in components:
        print('{}:{}(),'.format(node, components[node]))
    print('}}\nadj = {}\nmeasurement_nodes={}'.format(adj, measurement_nodes))


    plt.pause(0.002)
"""
