import scipy.optimize as opt
#from optimization.geneticalgorithminner import FIT_Inner
import numpy as np
"""
Using standard packages to slightly tweak the best of HOF to optimal individual
"""
def final_opt(optlist, ind, idx, env, experiment):    
    ind = unpack_opt(ind, optlist, idx)
    env.reset()
    experiment.setattributes(ind)
    experiment.simulate(env)
    
    fitness = env.fitness()
    return -fitness[0] * fitness[1]

"""
Unpacks the list of parameters from the structure used in the GA to the GD
"""
def pack_opt(ind, experiment):
    idx = []
    optlist = []
    for i in range(experiment.n_components): 
        c = experiment.nodes[i]['info']
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
def finetune_individual(ind, env, experiment):
    env.reset()
    
    (optlist, idx) = pack_opt(ind, experiment)
#    optres = opt.minimize(final_opt, optlist, args=(ind, idx, env, experiment), method='CG', options={'maxiter':10000})  
    optres = opt.minimize(final_opt, optlist, args=(ind, idx, env, experiment), method='Nelder-Mead', options={'maxiter':100000})    
    
    ind = unpack_opt(ind, optres.x.tolist(), idx)
    return ind
