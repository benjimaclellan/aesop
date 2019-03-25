#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:16:50 2019

@author: benjamin
"""
 
import copy 
from random import shuffle, sample
import time
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import networkx as nx
import keyboard

from assets.functions import extractlogbook, save_experiment, load_experiment, splitindices
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from assets.classes import Experiment
plt.close("all")

POTENTIALS = {'splitters':[PowerSplitter],
              'nonsplitters':[Fiber, AWG, PhaseModulator, WaveShaper]}

env = PulseEnvironment(p = 2, q = 1, profile = 'cw')

components = (
    {
     'fiber0': Fiber(),
     'ps0':PowerSplitter(),
     'pm0':PhaseModulator(),
     'fiber1':Fiber(),
     'ps1':PowerSplitter(),
     'fiber2':Fiber(),
    }
    ) 
adj = [('fiber0','ps0'), ('ps0','pm0'), ('pm0','ps1'), ('ps0','fiber1'), ('fiber1','ps1'), ('ps1', 'fiber2')]
measurement_nodes = ['fiber2']
#
#components = (
#        {
#         'awg':AWG(),
#         'fiber':Fiber(),
#        })
#adj = [('awg','fiber')]
#measurement_nodes = [1]


# now let's create the experiment as a custom class, build it based on specifications, and ensure it is properly connected
E = Experiment()
E.buildexperiment(components, adj, measurement_nodes)
#E.checkexperiment()

#E = Experiment()
#E,_ = brand_new_experiment(E)


def random_choice(a, N, replace=False):
    """
        Random subset from a list (with/without replacement). This supports lists containing multiple datatypes (unlike np.random.choice)
    """
    assert N > 0
    
    inds = np.random.choice(range(len(a)), N, replace=replace)
    choice = []
    for ind in inds:
        choice.append(a[ind])
    return choice


def get_nonsplitters(E):
    valid_nodes = []
    for node in E.nodes():
        if not E.nodes[node]['info'].splitter: #len(E.suc(node)) == 1 and len(E.pre(node)) == 1:
            valid_nodes.append(node)
    return valid_nodes

def select_N_new_components(N, comp_type, replace=False):
    """
        Selects N new components (power or freq), making a new instance of each 
    """
    assert N > 0
        
    inds = np.random.choice(range(len(POTENTIALS[comp_type])), N, replace=replace)
    new_comps = []
    for ind in inds:
        new_comps.append(POTENTIALS[comp_type][ind]())
    
    if N == 1:
        return new_comps[0]
    else:
        return new_comps
    
def reset_all_instances():
    for key, item in POTENTIALS.items():
        for class_i in POTENTIALS[key]:
            class_i.resetinstances(class_i)
    return

##
# -----------------------------------------------------------      
##    


def mutate_experiment(E):
    """
        Switch the placements of some nodes, the structure remains the same    
    """
    
    valid_nodes = get_nonsplitters(E)
    if len(valid_nodes) <= 1: #list is empty
#        raise ValueError('No valid nodes')
        return E, False
        
    replace_list = copy.copy(valid_nodes)
    shuffle(replace_list)
              
#    idx = range(len(valid_nodes))
#    i1, i2 = sample(idx, 2)
#    replace_list[i1], replace_list[i2] = valid_nodes[i2], valid_nodes[i1]
    
    
    
#    changes = []
#    n_swap = 2
#    swap_nodes = random_choice(valid_nodes, n_swap, replace = False)
#    Ecopy = copy.deepcopy(E)
#    
#    for i, node in enumerate(swap_nodes):
#        node_next = swap_nodes[(i+1)%n_swap]
#        for key in ('title', 'info'):
#            E.nodes[node][key] = Ecopy.nodes[node_next][key]
        
    changes = []
    for i in range(len(valid_nodes)):
        changes_i = {}
        if valid_nodes[i] != replace_list[i]:
            changes_i['node_original'] = valid_nodes[i]
            changes_i['node_shuffle'] = replace_list[i]
            
#            for key in E.nodes[changes_i['node_original']]:#('title', 'info'):
            for key in ('title', 'info'):
                changes_i[key] = E.nodes[valid_nodes[i]][key]
        
            changes.append(changes_i)
    mapping = {}
    for changes_i in changes:
        mapping[changes_i['node_original']] = changes_i['node_shuffle']
#    mapping = {changes_i['node_original']:changes_i['node_shuffle']}
    nx.relabel_nodes(E, mapping,copy=True)
    
    for changes_i in changes:    
#        for key in E.nodes[changes_i['node_original']]:
        for key in ('title', 'info'):
            E.nodes[changes_i['node_shuffle']][key] = changes_i[key]
#    
    return E, True
    
    
def mutate_new_components_experiment(E):
    """
        Changes a subset of components to brand new components (but not splitters)
    """
    
    valid_nodes = get_nonsplitters(E)
    if not valid_nodes: #list is empty
#        raise ValueError('No valid nodes')
        return E, False
        
        
    replace_list = random_choice(valid_nodes, 1, replace=False)

    for i, node in enumerate(replace_list):     
        E.nodes[node]['info'] = select_N_new_components(1, 'nonsplitters')
        E.nodes[node]['title'] = E.nodes[node]['info'].name
        
    return E, True
    

def one_component_to_two(E):
    """
        Adds in one new component where there is a component with one successor and predeccessor
    """

    valid_nodes = get_nonsplitters(E)
    if not valid_nodes: #list is empty
#        raise ValueError('No valid nodes')
        return E, False

    replace_list = random_choice(valid_nodes, 1, replace=False)
    
    for i, node in enumerate(replace_list):
        new_comp = select_N_new_components(1, 'nonsplitters')
        node_name = new_comp.name
        E.add_node(node_name)
        E.nodes[node_name]['title'] = node_name
        E.nodes[node_name]['info'] = new_comp
        
        before_after = np.random.random()
        if before_after >= 0.5: # add the new node BEFORE the current one
            if len(E.pre(node)) != 0: # if we have connections, delete them and rebuild
                pre = E.pre(node)[0]
                E.remove_edge(pre, node)
                E.add_edge(pre, node_name)
            E.add_edge(node_name, node)
        else: # add the new node AFTER the current one
            if len(E.suc(node)) != 0: # if we have connections, delete them and rebuild
                suc = E.suc(node)[0]
                E.remove_edge(node, suc)
                E.add_edge(node_name, suc)
            E.add_edge(node, node_name)
            
    return E, True

def remove_one_node(E):
    """
        Remove a single non-splitter node
    """    
    valid_nodes = get_nonsplitters(E)
    if len(valid_nodes) <= 1:
#        raise ValueError('There will be nothing left of the graph if we delete this node')
        return E, False
        
    node = random_choice(valid_nodes, 1)[0] # node to remove
    pre = E.pre(node)
    suc = E.suc(node)
    
    if len(pre) == 1:
        E.remove_edge(pre[0], node)
    if len(suc) == 1:
        E.remove_edge(node, suc[0])
    if len(pre) == 1 and len(suc) == 1:
        E.add_edge(pre[0], suc[0])    
    E.remove_node(node)
    return E, True
        
def add_loop(E):
    """
    Adds a interferometer(like) loop, replacing a single node 2 splitters/2 components
    """
    
    valid_nodes = get_nonsplitters(E)
    if not valid_nodes: #no valid nodes (list is empty)
#        raise ValueError('No valid nodes')
        return E, False    
    
    node = random_choice(valid_nodes, 1)[0]    
    
    new_comps = select_N_new_components(2, 'nonsplitters', replace = True)
    new_splitters = select_N_new_components(2, 'splitters', replace = True)
    
    ## add new splitters (can update later to have FrequencySplitters as well)
    for splitter_i in new_splitters:
        E.add_node(splitter_i.name)
        E.nodes[splitter_i.name]['title'] = splitter_i.name
        E.nodes[splitter_i.name]['info'] = splitter_i
    
    ## add new components and connect loop together
    for comp_i in new_comps:
        E.add_node(comp_i.name)
        E.nodes[comp_i.name]['title'] = comp_i.name
        E.nodes[comp_i.name]['info'] = comp_i
        
        E.add_edge(new_splitters[0].name, comp_i.name)
        E.add_edge(comp_i.name, new_splitters[1].name)
    
    pre = E.pre(node)
    suc = E.suc(node)
    
    if len(pre) == 1:
        E.remove_edge(pre[0], node)
        E.add_edge(pre[0], new_splitters[0].name)
        
    if len(E.suc(node)) == 1:
        E.remove_edge(node, suc[0])
        E.add_edge(new_splitters[1].name, suc[0])
    
    E.remove_node(node)
    
    return E, True

def remove_loop(E):
    undirE = E.to_undirected()
    cycles = nx.cycle_basis(undirE)
    valid_nodes = get_nonsplitters(E)
    
    if len(cycles) == 0:
        # there are no cycles
#        raise ValueError('No loops to remove')
        return E, False
        
    cycle_lens = []
    for cycle in cycles:
        cycle_lens.append(len(cycle)) 
    
    min_len = min(cycle_lens)
    if len(valid_nodes) < min_len + 1:
#        raise ValueError('There will be nothing left of the graph if we delete this loop')
        return E, False
    
    
    valid_cycles = []
    for cycle in cycles:
        if len(cycle) == min_len:
            valid_cycles.append(cycle)
    remove_cycle = random_choice(valid_cycles, 1)[0]
    for node in remove_cycle:    
        if not set(E.pre(node)).issubset(remove_cycle) or len(E.pre(node)) == 0:
            loop_init = node  
        if not set(E.suc(node)).issubset(remove_cycle) or len(E.suc(node)) == 0:
            loop_term = node

    ## patch graph
    if len(E.pre(loop_init)) == 1 and len(E.suc(loop_term)) == 1:
        E.add_edge(E.pre(loop_init)[0], E.suc(loop_term)[0])

    for node in remove_cycle:
        E.remove_node(node)
    
    return E, True

##
# -----------------------------------------------------------      
##    

def brand_new_experiment(E):
    reset_all_instances()
    
    start_comp = select_N_new_components(1, 'nonsplitters')
    components = {start_comp.name:start_comp}
    adj = []
    measurement_nodes = []
    E = Experiment()
    E.buildexperiment(components, adj, measurement_nodes)
    return E, True


##
# -----------------------------------------------------------      
##  
    



def change_experiment_wrapper(E):
    rates_base = {
                    mutate_experiment:0.4,
                    mutate_new_components_experiment:0.4,
                    one_component_to_two:0.3,
                    remove_one_node:0.3,
                    add_loop:0.1,
                    remove_loop:0.1,
                    brand_new_experiment:0.1
                 }

    rates = rates_base
    funcs = list(rates.keys())
    
    flag = True
    cnt = 0
    while flag:
        if E.number_of_nodes() > 10:
            rates[remove_one_node] = 0.9
            rates[remove_loop] = 0.9
            rates[one_component_to_two] = 0.1
            rates[add_loop] = 0.0
            rates[brand_new_experiment] = 0.4
            
        elif E.number_of_nodes() > 30:
            rates[brand_new_experiment] = 99
            
        elif E.number_of_nodes() < 3:
            rates[remove_one_node] = 0.1
            rates[remove_loop] = 0.05
            rates[one_component_to_two] = 0.4
            rates[add_loop] = 0.2
        
        else:
            rates = rates_base
        
        
        probs = list(rates.values())
        norm = sum(probs)
        for i,prob in enumerate(probs): probs[i] = prob/norm
        
        
        mut_function = np.random.choice(funcs, 1, replace=False, p=probs)[0]
#        E = mut_function(E)
        flag = False
#        print('Applying {} worked'.format(mut_function.__name__))
        
#        try:
        E, check = mut_function(E)
        if check:
#            print('There was a goo job up in ', mut_function.__name__)
            flag = False
        else:
#            print('There was a fuck up in ', mut_function.__name__)
            flag = True
        
        cnt += 1
        if cnt > 100:
            raise ValueError('There is something wrong with mutating the graph. An error occurs for everytime we try to change the graph structure')

    return E


##
# -----------------------------------------------------------      
##    

fig1 = plt.figure()
fig2 = plt.figure()

for i in range(200):
    for j in range(15):
        E = change_experiment_wrapper(E)
    E.cleanexperiment()
    E.make_path()
#    print(E.path)
    E.check_path()
    
    mapping=dict(zip( [item for sublist in E.path for item in sublist],range(0,len(E.nodes()))))
    E = nx.relabel_nodes(E, mapping) 
    
    E.checkexperiment()   

    plt.figure(fig1.number)
    plt.clf()
    E.draw(node_label='keys', title='its working')#, fig=fig1)
    
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
    E.measure(env, E.measurement_nodes[0], fig=fig2)  
    plt.show()



    plt.pause(0.1)


"""
save_experiment('test4', E)


E.draw(node_label='titles')

for node in E.nodes():
    info = E.nodes[node]['info']
    print('Name {}, title {}, splitter {}, class {}'.format(node, info.name, info.splitter, info.__class__.__name__) )

"""  

