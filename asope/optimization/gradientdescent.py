import scipy.optimize as opt
from copy import copy

"""
Wrapping function for final fine-tuning of the GA HOF individuals
"""
def finetune_individual(ind, env, experiment):
    (optlist, idx, nodelist) = pack_opt(ind, experiment)
    
    optres = opt.minimize(final_opt, optlist, args=(ind, idx, nodelist, env, experiment), method='Nelder-Mead', options={'maxiter':2000})    
    
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
    
    At = experiment.nodes[measurement_node]['output'].reshape(env.N)
    fitness = env.fitness(At)
    
    return -fitness[0] * fitness[1] #minus sign is because we want to minimize

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
            nodelist.append(node)
            cnt += 1
        a[1] = cnt
        idx.append(copy(a))
    return optlist, idx, nodelist

"""
Unpacks the list of parameters from the structure used in GD to the GA and simulation
"""
def unpack_opt(ind, experiment, optlist, idx, nodelist):
    for i, ind_pair in enumerate(idx, 0):
        start = ind_pair[0]
        end = ind_pair[1] + 1
        node = nodelist[i]
        cnt = 0
        for j in range(start, end):
            if (experiment.nodes[node]['info'].FINETUNE_SKIP != None) and (i-start) in (experiment.nodes[node]['info'].FINETUNE_SKIP):
                continue
            else:
                ind[node][cnt-start] = optlist[cnt]
                cnt += 1
    return ind
        

