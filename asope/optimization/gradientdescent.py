import scipy.optimize as opt
import autograd.numpy as np
from autograd import elementwise_grad
from copy import copy

def gradient_descent(at, env, exp, alpha=0.05, max_steps=2000):
    def analysis_wrapper(x_opt, at, env, exp, node_lst, idx_lst):
        exp.inject_optical_field(env.field)
        at = exp.attributes_from_list(x_opt, node_lst, idx_lst)
        exp.setattributes(at)
        exp.simulate(env)
        field = exp.nodes[exp.measurement_nodes[0]]['output']
        fit = env.fitness(field)
        return fit[0]

    x_opt, node_lst, idx_lst, sigma_lst, mu_lst, at_names = exp.experiment_info_as_list(at)
    func = lambda x: analysis_wrapper(x, at=at, env=env, exp=exp, node_lst=node_lst, idx_lst=idx_lst)

    grad = elementwise_grad(func)

    x = np.array(x_opt)
    for i_step in range(max_steps):
        fit = func(x)
        step = alpha * grad(x) / i_step
        x = x + step
        print('i_step: {}, L1_step: {}, fit: {}'.format(i_step, np.sum(np.power(step, 2)), fit))
        if np.sum(np.power(step, 2)) < 1e-8:
            break

    at = exp.attributes_from_list(x, at, node_lst, idx_lst)
    return at

#TODO: improve autodiff gradient descent
"""
    Below is all old code with used numerical packages for gradient descent. Now uses autograd
"""


"""
Wrapping function for final fine-tuning of the GA HOF individuals
"""
def finetune_individual(ind, env, experiment):
    (optlist, idx, nodelist) = pack_opt(ind, experiment)

    # extract the bounds into the correct data format
    lower_bounds, upper_bounds = {}, {}
    for node in ind.keys():
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

    field = experiment.nodes[measurement_node]['output']#.reshape(env.N)
    fitness = env.fitness(field)

    return -fitness[0]

"""
Unpacks the list of parameters from the structure used in the GA to the GD
"""
def pack_opt(ind, experiment):
    (optlist, idx, nodelist) = ([],[],[])
    cnt = 0
    a = [0,0] # range
    for node, item in ind.items():
        a[0] = cnt
        for i, at_i in enumerate(item, 0):
            if (experiment.nodes[node]['info'].FINETUNE_SKIP is not None) and (i in experiment.nodes[node]['info'].FINETUNE_SKIP): # paramter that should be skipped in finetuning
                pass
            else:
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
    for node_i, ind_pair in enumerate(idx, 0):
        node = nodelist[node_i]
        for j in range(0, len(ind[node])):
            if (experiment.nodes[node]['info'].FINETUNE_SKIP is not None) and (j in (experiment.nodes[node]['info'].FINETUNE_SKIP)):
                pass
            else:
                ind[node][j] = optlist[cnt]
            cnt += 1
    return ind


