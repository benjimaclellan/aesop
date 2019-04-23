

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
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from assets.classes import Experiment

from assets.graph_manipulation import *

plt.close("all")


#%%
env = PulseEnvironment(p = 2, q = 1, profile = 'cw')

components = {
1:FrequencySplitter(),
0:FrequencySplitter(),}
adj = [(0, 1), (0, 1) ]
measurement_nodes=[1]

#at = {2: [1, 1e7, 1]}

E = Experiment()
E.buildexperiment(components, adj, measurement_nodes)
E.checkexperiment()

E.make_path()
E.check_path()

E.draw(node_label='disp_name')

for node in E.nodes():
    if not E.pre(node):
        E.nodes[node]['input'] = env.At
        
at = E.newattributes()
E.setattributes(at)

E.simulate(env)

E.measure(env, measurement_nodes[0], check_power=True)
plt.show()



#%%

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
    save_experiment('test4', E)
    
    
    E.draw(node_label='titles')
    
    for node in E.nodes():
        info = E.nodes[node]['info']
        print('Name {}, title {}, splitter {}, class {}'.format(node, info.name, info.splitter, info.__class__.__name__) )
    
    """  
