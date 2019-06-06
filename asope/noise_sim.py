import numpy as np
from assets.graph_manipulation import get_nonsplitters
import scipy.integrate as integrate
from scipy.special import binom
from copy import deepcopy

def update_error_attributes(experiment):
    """
    Goes through each component in the experiment and gives them new sampled error parameters

    :param experiment:
    :return:
    """
    exp = experiment
    for node in exp.node():
        # Check that there is an error model to update
        try:
            if exp.node()[node]['info'].N_EPARAMETERS != 0:
                component = exp.node()[node]['info']
                #error_model() will return an array of appropriately sampled error attributes
                #update_error_attributes() will use the array to save the properties to the component
                new_at = component.error_model()

                component.update_error_attributes(new_at)
        except AttributeError:
            print(node)
            print("Exception occurred updating error models - is there a component without an implemented error model?")


def simulate_component_noise(experiment, environment, input_At, N_samples):
    """
    Runs a monte carlo simulation of the component noise. Returns a (N_samples, m) array, where m is the number
    of parameters returned by environment.fitness

    :param experiment:
    :param environment:
    :param input_At:
    :param N_samples:
    :return:
    """
    #rename variables
    At = input_At
    env = environment
    exp = experiment
    m = np.shape(env.fitness(At))[0]
    #create empty array with the correct shape
    fitnesses = np.zeros((N_samples, m))
    length = np.shape(At)[0]
    optical_fields = np.zeros((N_samples, length), dtype=np.complex)
    for i in np.arange(N_samples):
        update_error_attributes(exp)
        # Simulate the experiment with the sampled error profile
        exp.simulate(env)
        At = exp.nodes[exp.measurement_nodes[0]]['output']
        optical_fields[i] = At[:, 0]
        # Evaluate the fitness and store it in the array
        fit = env.fitness(At)
        fitnesses[i] = fit

    return fitnesses, optical_fields

def drop_node(experiment, node):
    """
        Remove a specific single node

    :param experiment:
    :param node:
    """
#    print('remove_one_node')

    pre = experiment.pre(node)
    suc = experiment.suc(node)

    if len(pre) == 1:
        experiment.remove_edge(pre[0], node)
    if len(suc) == 1:
        experiment.remove_edge(node, suc[0])
    if len(pre) == 1 and len(suc) == 1:
        experiment.add_edge(pre[0], suc[0])
    experiment.remove_node(node)
    return experiment

def remove_redundancies(experiment, env, verbose=False):
    """
    Check over an experiment for redundant nodes

    :param experiment:
    :param env:
    :param verbose:
    :return:
    """
    exp = experiment
    # Compute the fitness of the unmodified experiment, for comparison
    At = exp.nodes[exp.measurement_nodes[0]]['output']
    fit = env.fitness(At)
    valid_nodes = get_nonsplitters(exp)

    for node in valid_nodes[:]:
        if verbose:
            print("Dropping node: " + str(node))
        exp_bak = deepcopy(exp)
        exp_mod = drop_node(exp_bak, node) # Drop a node from the experiment
        exp_mod.measurement_nodes = exp_mod.find_measurement_nodes()
        exp_mod.checkexperiment()
        exp_mod.make_path()
        exp_mod.check_path()
        exp_mod.inject_optical_field(env.At)
        # Simulate the optical table with dropped node
        exp_mod.simulate(env)
        At_mod = exp_mod.nodes[exp_mod.measurement_nodes[0]]['output']
        fit_mod = env.fitness(At_mod)
        if verbose:
            print("Obtained fitness of " + str(fit_mod))
            print("-_-_-_-")
        # Compare the fitness to the original optical table
        if fit_mod[0] >= fit[0]: #and fit_mod[1] >= fit[1]:
            print("Dropping node")
            # If dropping the node didn't hurt the fitness, then drop it permanently
            exp = exp_mod
            fit = fit_mod
            At = At_mod

    return exp

def get_error_parameters(experiment):
    """
    Iterate through the list of nodes in the experiment, and append their error keyvals into an array.
    The components of the array are tuples of the form [node, key] which has the dictionary key 'node'
    indicating which node and 'key' indicating which error parameter.

    :param experiment:
    :return:
    """
    error_params = np.zeros((0,2))
    for node in experiment.nodes():
        node_ep = experiment.nodes()[node]['info'].ERROR_PARAMETERS
        for key in node_ep:
            tuple = np.array([[node, key]])
            error_params = np.append(error_params, tuple, axis=0)

    return error_params

def get_error_functions(experiment):
    """
    Iterature through the list of nodes in the experiment, and append their error pdf keyvals into an array.

    :param experiment:
    :return:
    """
    error_functions = np.zeros(0)
    for node in experiment.nodes():
        node_ep = experiment.nodes()[node]['info'].ERROR_PDFS
        for key in node_ep:
            tuple = [node, key]
            fx = node_ep[key]
            error_functions = np.append(error_functions, fx)

    return error_functions


def simulate_with_error(perturb, experiment, environment):
    """
    Given an error perturbation, return the fitness.
    Param should be a triple of the form [node, key, val] where node and key are dictionary keys identifying
    which error parameter is being perturbed, and val is the value of the perturbation.

    :param perturb:
    :param experiment:
    :param environment:
    :return:
    """
    # Perturb is a triple with the error parameter key and the new value
    if perturb is not None:
        node, key, val = perturb
        experiment.nodes()[int(node)]['info'].ERROR_PARAMETERS[key] = val

    experiment.simulate(environment)
    At = experiment.nodes[experiment.measurement_nodes[0]]['output']

    fit = environment.fitness(At)
    return fit[0]

'''
j indexes the random variable i.e. j \in [1,N]
i indexes the moment i.e. i \in [1,l]

'''

def normal_pdf(x, mu=0, sigma=1):
    """
    Probability distribution function of normal distribution.

    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    scale = 1/(np.sqrt(2*np.pi*sigma**2))
    exp = np.exp(-1*(x-mu)**2/(2*sigma**2))
    return scale*exp

def evintegrand(val,node,key,fx,y,m):
    """
    Returns the integrand for the expected value function. Meant to be called by expectationValueM.

    :param val:
    :param node:
    :param key:
    :param fx:
    :param y:
    :param m:
    :return:
    """
    perturb = [node, key, val]
    return y(perturb)**m*fx(val)

def expectationValueM(node, key, fx, y, m):
    """
    Compute the expectation value of Y^m (mu_i,...,x_j,...,mu_N) for a given error parameter x_j and function y.

    :param node:
    :param key:
    :param fx:
    :param y:
    :param m:
    :return:
    """
    I = integrate.quad(evintegrand, -np.inf, np.inf, args=(node, key, fx, y, m))
    return I[0]

def S(i,j,y, error_params, error_functions):
    """
    Recursive function to compute Sij required for UDR, returns float
    Sij = Sum(k=0...i) iCk Skj-1 E(Y^(i-k)(mu1,...,Xj,...,muN)
    """
    # Get the jth error parameter
    node, key = error_params[j-1]
    fx = error_functions[j-1]

    if j<1:
        # j should never be negative
        raise AttributeError
    if j==1:
        a = expectationValueM(node, key, fx, y, i)
        return a
    else:
        sum = 0
        for k in np.arange(i+1):
            sum += binom(i, k)*expectationValueM(node, key, fx, y, i-k)*S(k, j-1, y, error_params, error_functions)
        return sum

def UDR_moments(y, l, error_params, error_functions):
    """
    Compute the lth moment of Y(error_params) using univariate dimension reduction.
    See 'A univariate dimension-reduction method for multi-dimensional integration in stochastic mechanics' by Rahman and Xu

    :param y:
    :param l:
    :param error_params:
    :param error_functions:
    :return:
    """
    N = np.shape(error_params)[0] # Get the number of error params
    moment = 0.
    meanval = y(['0','phasenoise',0]) #TODO: properly get the meanval
    for i in np.arange(l+1):
        sval = S(i, N, y, error_params, error_functions) # Compute S^i_N
        moment += binom(l, i)*sval*(-(N-1)*meanval)**(l-i)
    return moment

def compute_moment_matrix(error_params, error_functions, M):
    N = np.shape(error_params)[0]
    moment_matrix = np.zeros([N,M])
    identity = lambda x: x
    for j in np.arange(N):
        node, key = error_params[j-1]
        fx = error_functions[j-1]
        for i in np.arange(M):
            mu_ij = expectationValueM(node, key, fx, identity, i)
            moment_matrix[j, i] = mu_ij

    return moment_matrix
