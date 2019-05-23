import scipy.optimize as opt
from copy import copy
import numpy as np


"""
Wrapping function for final fine-tuning of the GA HOF individuals
"""
def finetune_individual(ind, env, experiment):
    (optlist, idx, nodelist) = pack_opt(ind, experiment)
    
    # extract the bounds into the correct data format
    lower_bounds, upper_bounds = {}, {}
    for node in experiment.nodes():
        lower_bounds[node] = experiment.nodes[node]['info'].LOWER
        upper_bounds[node] = experiment.nodes[node]['info'].UPPER
    (lower, idx, nodelist) = pack_opt(lower_bounds, experiment)
    (upper, idx, nodelist) = pack_opt(upper_bounds, experiment)
    bounds = list(zip(lower, upper))
    
    # the Nelder-Mead method seems to be faster, but does not allow bounded
#    optres = opt.minimize(final_opt, optlist, args=(ind, idx, nodelist, env, experiment), method='Nelder-Mead', options={'maxiter':1000}, adaptive=True) 
    
    # we use the L-BFGS-B algorithm as it allows bounded optimization
    optres = opt.minimize(final_opt, optlist, args=(ind, idx, nodelist, env, experiment), method='L-BFGS-B', bounds=bounds, options={'maxiter':100})  
    
    # now unpack all fine-tuned values
    ind = unpack_opt(ind, experiment, optres.x.tolist(), idx, nodelist)
    return ind

"""
Using standard packages to slightly tweak the best of HOF to optimal individual
"""
def final_opt(optlist, ind, idx, keylist, env, experiment):    
    ind = unpack_opt(ind, experiment, optlist, idx, keylist)
    
    measurement_node = experiment.measurement_nodes[0]
    
    experiment.setattributes(ind)
    experiment.simulate(env)
    
    At = experiment.nodes[measurement_node]['output']#.reshape(env.N)
    fitness = env.fitness(At)
    
    return -fitness[0]

"""
Unpacks the list of parameters from the structure used in the GA to the GD
"""
def pack_opt(ind, experiment):
    (optlist, idx, nodelist) = ([],[],[])
    cnt = 0
    a = [0,0]
    for node, item in ind.items():
        a[0] = cnt
        for i, at_i in enumerate(item, 0):
            if (experiment.nodes[node]['info'].FINETUNE_SKIP != None) and i in (experiment.nodes[node]['info'].FINETUNE_SKIP):
                continue
            optlist.append(at_i)
        
            cnt += 1
        nodelist.append(node)
        a[1] = cnt
        idx.append(copy(a))

    return optlist, idx, nodelist

"""
Unpacks the list of parameters from the structure used in GD to the GA and simulation
"""
def unpack_opt(ind, experiment, optlist, idx, nodelist):
    cnt = 0
    for i, ind_pair in enumerate(idx, 0):
        node = nodelist[i]
        for j in range(0, len(ind[node])):
            if (experiment.nodes[node]['info'].FINETUNE_SKIP != None) and (j) in (experiment.nodes[node]['info'].FINETUNE_SKIP):
                pass
            else:
                ind[node][j] = optlist[cnt]
                cnt += 1
    return ind
        

