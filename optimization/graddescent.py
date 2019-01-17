import scipy.optimize as opt
from optimization.ga_functions_inner import FIT_Inner

"""
Using standard packages to slightly tweak the best of HOF to optimal individual
"""
def final_opt(optlist, ind, idx, gap, env, experiment, sim):    
    ind = unpack_opt(ind, optlist, idx)
    fitness = -FIT_Inner(ind, gap, env, experiment, sim)[1]
    return fitness

"""
Unpacks the list of parameters from the structure used in the GA to the GD
"""
def pack_opt(ind, experiment):
    idx = []
    optlist = []
    for i in range(len(experiment)): 
        c = experiment[i]
        for j in range(c.FINETUNE_SKIP, len(ind[i])):
            idx.append( [ i, j ] )
            optlist.append(ind[i][j])
    return optlist, idx


"""
Unpacks the list of parameters from the structure used in GD to the GA and simulation
"""
def unpack_opt(ind, optlist, idx):
    for i in range(len(idx)):
        ind[idx[i][0]][idx[i][1]] = optlist[i]
    return ind
        
"""
Wrapping function for final fine-tuning of the GA HOF individuals
"""
def finetune_individual(ind, gap, env, experiment, sim):
    (optlist, idx) = pack_opt(ind, experiment)
    
    optres = opt.minimize(final_opt, optlist, args=(ind, idx, gap, env, experiment, sim), method='CG', options={'maxiter':10000})    
    
    ind = unpack_opt(ind, optres.x.tolist(), idx)
    return ind
