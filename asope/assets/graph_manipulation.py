"""
Copyright Benjamin MacLellan

Graph manipulation functions. These change the topology of the graph based on physically realistic rules.
"""
#%% library imports
import copy 
from random import shuffle
import numpy as np
import networkx as nx

#%% custom module imports
from classes.experiment import Experiment

#%% 
def max_int_in_list(lst):
    """
        Finds the maximum integer in a list of mixed datatypes, used to label new nodes
    """
    return max((i for i in lst if isinstance(i, int)))

#%%
def random_choice(options, num_choices, replace=False):
    """
        Random subset from a list (with/without replacement). This supports lists containing multiple datatypes (unlike np.random.choice)
    """
    assert num_choices > 0
    inds = np.random.choice(range(len(options)), num_choices, replace=replace)
    choice = []
    for ind in inds:
        choice.append(options[ind])
    return choice

#%%
def get_nonsplitters(experiment):
    valid_nodes = []
    for node in experiment.nodes():
        if not experiment.nodes[node]['info'].splitter: 
            valid_nodes.append(node)
    return valid_nodes

#%%
def select_N_new_components(num_new_components, POTENTIAL_COMPS, comp_type, replace=False):
    """
        Selects N new components (power or freq), making a new instance of each 
    """
    assert num_new_components > 0
        
    inds = np.random.choice(range(len(POTENTIAL_COMPS[comp_type])), num_new_components, replace=replace)
    new_comps = []
    for ind in inds:
        new_comps.append(POTENTIAL_COMPS[comp_type][ind]())
    
    if num_new_components == 1:
        return new_comps[0]
    else:
        return new_comps
    
    
#%%
def reset_all_instances(POTENTIAL_COMPS):    
    for key, item in POTENTIAL_COMPS.items():
        for class_i in POTENTIAL_COMPS[key]:
            class_i.resetinstances(class_i)
    return

#%%  
def mutate_experiment(experiment, POTENTIAL_COMPS):
    """
        Switch the placements of some nodes, the structure remains the same    
    """
#    print('mutate_experiment')
    valid_nodes = get_nonsplitters(experiment)
    if len(valid_nodes) <= 1: #list is empty
        return experiment, False
        
    replace_list = copy.copy(valid_nodes)
    shuffle(replace_list)

    changes = []
    for i in range(len(valid_nodes)):
        changes_i = {}
        if valid_nodes[i] != replace_list[i]:
            changes_i['node_original'] = valid_nodes[i]
            changes_i['node_shuffle'] = replace_list[i]
            
            for key in ('title', 'info'):
                changes_i[key] = experiment.nodes[valid_nodes[i]][key]
        
            changes.append(changes_i)
    mapping = {}
    for changes_i in changes:
        mapping[changes_i['node_original']] = changes_i['node_shuffle']
    nx.relabel_nodes(experiment, mapping, copy=True)
    
    for changes_i in changes:    
        for key in ('title', 'info'):
            experiment.nodes[changes_i['node_shuffle']][key] = changes_i[key]
    
    return experiment, True
    
#%%   
def mutate_new_components_experiment(experiment, POTENTIAL_COMPS):
    """
        Changes a subset of components to brand new components (but not splitters)
    """
#    print('mutate_new_components_experiment')
    valid_nodes = get_nonsplitters(experiment)
    if not valid_nodes: #list is empty
        return experiment, False
        
    replace_list = random_choice(valid_nodes, 1, replace=False)
    
    for i, node in enumerate(replace_list):     
        experiment.nodes[node]['info'] = select_N_new_components(1, POTENTIAL_COMPS, 'nonsplitters')
        experiment.nodes[node]['title'] = experiment.nodes[node]['info'].name
        nx.relabel_nodes(experiment, {node:max_int_in_list(experiment.nodes)+1}, copy=True)
        
    return experiment, True
    
#%%
def one_component_to_two(experiment, POTENTIAL_COMPS):
    """
        Adds in one new component where there is a component with one successor and predeccessor
    """
#    print('one_component_to_two')
    valid_nodes = get_nonsplitters(experiment)
    if not valid_nodes: #list is empty
        return experiment, False

    replace_list = random_choice(valid_nodes, 1, replace=False)
    
    for i, node in enumerate(replace_list):
        new_comp = select_N_new_components(1, POTENTIAL_COMPS, 'nonsplitters')
        node_name = max_int_in_list(experiment.nodes) + 1
        experiment.add_node(node_name)
        experiment.nodes[node_name]['title'] = new_comp.name
        experiment.nodes[node_name]['info'] = new_comp
        
        before_after = np.random.random()
        if before_after >= 0.5: # add the new node BEFORE the current one
            if len(experiment.pre(node)) != 0: # if we have connections, delete them and rebuild
                pre = experiment.pre(node)[0]
                experiment.remove_edge(pre, node)
                experiment.add_edge(pre, node_name)
            experiment.add_edge(node_name, node)
        else: # add the new node AFTER the current one
            if len(experiment.suc(node)) != 0: # if we have connections, delete them and rebuild
                suc = experiment.suc(node)[0]
                experiment.remove_edge(node, suc)
                experiment.add_edge(node_name, suc)
            experiment.add_edge(node, node_name)
            
    return experiment, True

#%%
def remove_one_node(experiment, POTENTIAL_COMPS):
    """
        Remove a single non-splitter node
    """    
    
#    print('remove_one_node')
    
    valid_nodes = get_nonsplitters(experiment)
    if len(valid_nodes) <= 1:
#        raise ValueError('There will be nothing left of the graph if we delete this node')
        return experiment, False
        
    node = random_choice(valid_nodes, 1)[0] # node to remove
    pre = experiment.pre(node)
    suc = experiment.suc(node)
    
    if len(pre) == 1:
        experiment.remove_edge(pre[0], node)
    if len(suc) == 1:
        experiment.remove_edge(node, suc[0])
    if len(pre) == 1 and len(suc) == 1:
        experiment.add_edge(pre[0], suc[0])    
    experiment.remove_node(node)
    return experiment, True
  
#%%      
def add_loop(experiment, POTENTIAL_COMPS):
    """
    Adds a interferometer(like) loop, replacing a single node 2 splitters/2 components
    """
#    print('***************add_loop')
    if not POTENTIAL_COMPS['splitters']:
        return experiment, True
    
    valid_nodes = get_nonsplitters(experiment)
    if not valid_nodes: #no valid nodes (list is empty)
        return experiment, False    
    
    node = random_choice(valid_nodes, 1)[0]    
    
    tmp = max_int_in_list(experiment.nodes)+1
    
    new_comps = select_N_new_components(2, POTENTIAL_COMPS, 'nonsplitters', replace = True)
    new_comps_names = [tmp+1, tmp+2]
    
    new_splitters = select_N_new_components(2, POTENTIAL_COMPS, 'splitters', replace = True)
    new_splitters_names = [tmp+3, tmp+4]
    
    
    ## add new splitters (can update later to have FrequencySplitters as well)
    for i, splitter_i in enumerate(new_splitters):
        experiment.add_node(new_splitters_names[i])
        experiment.nodes[new_splitters_names[i]]['title'] = splitter_i.name
        experiment.nodes[new_splitters_names[i]]['info'] = splitter_i
    
    ## add new components and connect loop together
    for i, comp_i in enumerate(new_comps):
        experiment.add_node(new_comps_names[i])
        experiment.nodes[new_comps_names[i]]['title'] = comp_i.name
        experiment.nodes[new_comps_names[i]]['info'] = comp_i
        
        experiment.add_edge(new_splitters_names[0], new_comps_names[i])
        experiment.add_edge(new_comps_names[i], new_splitters_names[1])
    
    pre = experiment.pre(node)
    suc = experiment.suc(node)
    
    if len(pre) == 1:
        experiment.remove_edge(pre[0], node)
        experiment.add_edge(pre[0], new_splitters_names[0])
        
    if len(experiment.suc(node)) == 1:
        experiment.remove_edge(node, suc[0])
        experiment.add_edge(new_splitters_names[1], suc[0])
    
    experiment.remove_node(node)

    return experiment, True

#%%
def remove_loop(experiment,POTENTIAL_COMPS):
#    print('remove_loop')
    
    undirE = experiment.to_undirected()
    cycles = nx.cycle_basis(undirE)
    valid_nodes = get_nonsplitters(experiment)
    
    if len(cycles) == 0:
        # there are no cycles
#        raise ValueError('No loops to remove')
        return experiment, False
        
    cycle_lens = []
    for cycle in cycles:
        cycle_lens.append(len(cycle)) 
    
    min_len = min(cycle_lens)
    if len(valid_nodes) < min_len + 1:
#        raise ValueError('There will be nothing left of the graph if we delete this loop')
        return experiment, False
    
    
    valid_cycles = []
    for cycle in cycles:
        if len(cycle) == min_len:
            valid_cycles.append(cycle)
    remove_cycle = random_choice(valid_cycles, 1)[0]
    for node in remove_cycle:    
        if not set(experiment.pre(node)).issubset(remove_cycle) or len(experiment.pre(node)) == 0:
            loop_init = node  
        if not set(experiment.suc(node)).issubset(remove_cycle) or len(experiment.suc(node)) == 0:
            loop_term = node

    ## patch graph
    if len(experiment.pre(loop_init)) == 1 and len(experiment.suc(loop_term)) == 1:
        experiment.add_edge(experiment.pre(loop_init)[0], experiment.suc(loop_term)[0])

    for node in remove_cycle:
        experiment.remove_node(node)
    
    return experiment, True
 
#%%
def brand_new_experiment(experiment, POTENTIAL_COMPS):
#    print('brand_new_experiment')
    
    reset_all_instances(POTENTIAL_COMPS)
    
    start_comp = select_N_new_components(1, POTENTIAL_COMPS,  'nonsplitters')
    components = {0:start_comp}
    adj = []
    measurement_nodes = [0]
    experiment = Experiment()
    experiment.buildexperiment(components, adj, measurement_nodes)
    
    return experiment, True

#%%
def change_experiment_wrapper(experiment, POTENTIAL_COMPS):
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
        if experiment.number_of_nodes() > 7:
            rates[remove_one_node] = 0.9
            rates[remove_loop] = 0.9
            rates[one_component_to_two] = 0.1
            rates[add_loop] = 0.0
            rates[brand_new_experiment] = 0.4
            
        elif experiment.number_of_nodes() > 10:
            rates[brand_new_experiment] = 999
            
        elif experiment.number_of_nodes() < 2:
            rates[remove_one_node] = 0.1
            rates[remove_loop] = 0.05
            rates[one_component_to_two] = 0.4
            rates[add_loop] = 0.2
        else:
            rates = rates_base
        
        
        
        num_splitters = 0
        for node in experiment.nodes():
            if experiment.nodes[node]['info'].splitter:
                num_splitters += 1
        if num_splitters/experiment.number_of_nodes() > 0.6:
            rates[remove_one_node] = 0.9
            rates[remove_loop] = 0.9
            rates[one_component_to_two] = 0.0
            rates[add_loop] = 0.0
            rates[brand_new_experiment] = 10
            
        
        probs = list(rates.values())
        norm = sum(probs)
        for i,prob in enumerate(probs): probs[i] = prob/norm
        
        
        mut_function = np.random.choice(funcs, 1, replace=False, p=probs)[0]
        flag = False
        
        experiment, check = mut_function(experiment, POTENTIAL_COMPS)
        if check:
            flag = False
        else:
            flag = True
        
        cnt += 1
        if cnt > 100:
            raise ValueError('There is something wrong with mutating the graph. An error occurs for everytime we try to change the graph structure')
    
    return experiment, flag



def remake_experiment(ind):
    
    ind.cleanexperiment()    
    mapping=dict(zip(ind.nodes(),range(0,len(ind.nodes()))))
    ind = nx.relabel_nodes(ind, mapping)  
    
    components = {}
    for node in ind.nodes():
        components[node] = ind.nodes[node]['info'].__class__()
#        if len(ind.suc(node)) == 0:
#            measurement_nodes.append(node)
#    print(measurement_nodes)
    adj = list(ind.edges())

    experiment = Experiment()
    experiment.buildexperiment(components, adj)#, measurement_nodes)    

    experiment.make_path()
    experiment.check_path()
    
    return experiment
