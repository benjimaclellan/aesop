import autograd.numpy as np
from autograd import grad
from assets.graph_manipulation import get_nonsplitters
import scipy.integrate as integrate
from scipy.special import binom
from scipy.misc import factorial2
from copy import deepcopy
import sys

def perturb_experiment_parameters(experiment):
    '''

    :param experiment:
    :param input_params:
    :param input_pdfs:
    :return:
    '''
    # Get the original set of attributes
    at_original = experiment.getattributes()
    # Get the list of nodes
    expnodes = experiment.nodes()

    # Sample the input parameters from their pdfs
    at_perturb = {}
    for node in expnodes:
        node_at = []
        for moments in expnodes[node]['info'].at_pdfs:
            mu, sigma = moments
            sample_val = np.random.normal(mu, sigma)

            node_at.append(sample_val)

        at_perturb[node] = node_at

    new_at = {node: [sum(x) for x in zip(at_original[node], at_perturb[node])] for node in at_original}

    experiment.setattributes(new_at)

    return new_at

def mc_error_propagation(experiment, environment, N):
    fitnesses = np.zeros(N)
    print("Monte Carlo simulation:")
    bar_length = 20

    at_optimal = experiment.getattributes()
    for i in np.arange(N):
        progress = int(bar_length * (i+1)/N)
        sys.stdout.write("\r" + "[" + "|"*progress + " "*(bar_length-progress) + "]")
        sys.stdout.flush()
        # Randomly perturb the experiment
        perturb_experiment_parameters(experiment)

        # Simulate the experiment with the sampled error profile
        experiment.simulate(environment)
        At = experiment.nodes[experiment.measurement_nodes[0]]['output']

        # Evaluate the fitness and store it in the array
        fitnesses[i] = environment.fitness(At)[0]

    print(" ")
    # Reset the experiment to optimal parameters
    experiment.setattributes(at_optimal)

    return fitnesses

def drop_node(experiment, node):
    """
        Remove a specific single node

    :param experiment:
    :param node:
    """

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
    error_params = np.zeros((0, 2))
    for node in experiment.nodes():
        i = 0
        node_at = experiment.node()[node]['info'].at
        for val in node_at:
            tuple = np.array([[node, i]])
            i += 1
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
        '''
        node_ep = experiment.nodes()[node]['info'].ERROR_PDFS
        for key in node_ep:
            fx = node_ep[key]
            error_functions = np.append(error_functions, fx)
        '''

        node_pdfs = experiment.node()[node]['info'].at_pdfs
        for fx in node_pdfs:
            error_functions = np.append(error_functions, fx)

    return error_functions


def simulate_with_error(perturb, experiment, environment):
    simulate_with_error.count += 1
    """
    Given an error perturbation, return the fitness.

    Param should be a triple of the form [node, key, val] where node and key are dictionary keys identifying
    which error parameter is being perturbed, and val is the value of the perturbation.

    Currently, this code will only work for one fitness parameter.

    :param perturb:
    :param experiment:
    :param environment:
    :return float:
    """
    # Perturb is a triple with the error parameter key and the new value
    node, index, val = perturb

    if index - int(index) != 0:
        raise ValueError("All attribute indices should be integers!")
    index = int(index)

    mu, sigma = experiment.node()[int(node)]['info'].at_pdfs[index]
    upper = experiment.node()[int(node)]['info'].UPPER[index]
    lower = experiment.node()[int(node)]['info'].LOWER[index]

    # get the current value
    optimal_val = experiment.nodes()[int(node)]['info'].at[index]

    # Perturb is a triple with the error parameter key and the new value
    # co-ordinate xform
    real_val = sigma*val + mu

    real_val += optimal_val

    if real_val > upper:
        real_val = upper
    if real_val < lower:
        real_val = lower

    experiment.nodes()[int(node)]['info'].at[index] = real_val

    experiment.simulate(environment)
    At = experiment.nodes[experiment.measurement_nodes[0]]['output']

    fit = environment.waveform_temporal_overlap(At)

    # reset to optimal value
    experiment.node()[int(node)]['info'].at[index] = optimal_val

    return fit[0]

simulate_with_error.count = 0

'''
j indexes the random variable i.e. j in [1,N]
i indexes the moment i.e. i in [1,l]

'''

def normal_pdf(x, mu=0, sigma=1):
    """
    Probability distribution function of normal distribution.

    :param x:
    :param mu:
    :param sigma:
    :return float:
    """
    scale = 1/(np.sqrt(2*np.pi*sigma**2))
    exp = np.exp(-1*(x-mu)**2/(2*sigma**2))
    return scale*exp

def evintegrand(val, node, key, fx, y, m):
    """
    Returns the integrand for the expected value function. Meant to be called by expectationValueM.

    :param val:
    :param node:
    :param key:
    :param fx:
    :param y:
    :param m:
    :return float:
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
    :return float:
    """
    expected_value = integrate.quad(evintegrand, -np.inf, np.inf, args=(node, key, fx, y, m))
    return expected_value[0]

'''
Functions from here down are used for UDR calculations. The conventions are as follows;
j in [1,...,N] indexes the input random variables
i in [1,...,n] indexes the interpolation points

for python arrays indexed starting from 0, we must always call the array element i-1 or j-1

UDR = univariate dimensional reduction. see Rahman and Xu (2004)
'''

def S(i, j, y, matrix_array, input_variables, input_pdfs, x):

    """
    Computes Sij, coefficients required for UDR. Recursive function that recurses on j.

    Sij = Sum(k=0...i) iCk Skj-1 E(Y^(i-k)(mu1,...,Xj,...,muN)

    :param i:
    :param j:
    :param y:
    :param matrix_array:
    :param input_variables:
    :param input_pdfs:
    :param x:
    :return float:
    """

    if j<1:
        # j should range from N to 1, never less
        raise ValueError("j should never be less then 1")

    if j==1:
        # The base case, Si1 = E[Y^i(X_1, mu_2, ... , mu_N)]
        base = UDR_evCalculation(j, y, i, x, matrix_array, input_variables)
        return base

    else:
        sigma = 0
        for k in np.arange(i+1): # Sum k=0 up to i
            # Compute S recursively
            S_val = S(k, j - 1, y, matrix_array, input_variables, input_pdfs, x)
            sigma += binom(i, k) * UDR_evCalculation(j, y, i - k, x, matrix_array, input_variables) * S_val
        return sigma

def UDR_moments(y, l, input_variables, input_pdfs, x, matrix_array):
    """
    Compute the lth moment of Y(error_params) using univariate dimension reduction.
    See 'A univariate dimension-reduction method for multi-dimensional integration in stochastic mechanics' 
    by Rahman and Xu (2004) for an exposition of the technique.

    :param y:
    :param l:
    :param input_variables:
    :param input_pdfs:
    :param x:
    :param matrix_array:
    :return float:
    """
    N = np.shape(input_variables)[0] # Get the number of input random variables
    moment = 0.

    for i in np.arange(l+1): # sum i = 0 to l
        sval = S(i, N, y, matrix_array, input_variables, input_pdfs, x) # Compute S^i_N
        moment += binom(l, i)*sval*(-(N-1)*0)**(l-i) #y(mu) = 0, due to shifting in compute_with_error

    return moment

def compute_moment_matrices(input_variables, n):
    """
    To find the interpolation points we need to construct a matrix

    |mu_(j, n-1), ... , (-1)^(n-1) mu_(j, 0) |
    |mu_(j, n)  , ... , (-1)^(n-1) mu_(j, 1) |
    |   ...     , ... ,         ...          |
    |mu(j, 2n-2), ... , (-1)^(n-1) mu_(j,n-1)|

    this function will return a 3d matrix, where the jth element is the above 2d matrix
    
    :param input_variables:
    :param input_pdfs:
    :param n:
    :return:
    """
    ## We want to solve Ax = b but need to find A and b
    matrix_array = np.zeros((0, n, n+1))
    moment_matrix = np.zeros([n, n+1])
    identity = lambda x: x[2] #TODO: remove jank

    for j in np.arange(1, np.shape(input_variables)[0] + 1): # j=1,..,N
        node, key = input_variables[j - 1]
        fx = normal_pdf # use a standard normal and do a co-ordinate transform later
        ## Here we construct an n x n+1 matrix, which we will split into a n dimensional column vector
        ## and a n x n dimensional matrix M (the moment matrix)
        for i in np.arange(1, n+2): # i = 1,...,n+1
            for k in np.arange(1, n+1): # k = 1,...,n
                mu_ik = ((-1)**i)*expectationValueM(node, key, fx, identity, n-i+k)
                if i == 1: # The 0th column should be positive as I am going to slice it off as b
                    mu_ik = -1*mu_ik
                moment_matrix[k-1, i-1] = mu_ik

        matrix_array = np.append(matrix_array, np.array([moment_matrix]), axis=0)

    return matrix_array

def compute_interpolation_points(matrix_array):
    """
    Invert the moment matrix to get polynomial coefficients and find the roots, which are the interpolation points
    for the UDR integral approximation.

    :param matrix_array:
    :return ndarray:
    """
    # declare arrays
    N = np.shape(matrix_array)[2]
    x = np.zeros((0, N-1))
    r = np.zeros((0, N))

    for j in np.arange(np.shape(matrix_array)[0]):
        # get the jth moment matrix
        moment_matrix = matrix_array[j-1]
        # slice off the b vector (Ax = b)
        b = moment_matrix[:, 0]
        A = moment_matrix[:, 1:]
        # solve for the interpolation points xi
        # 0 = x0^N + r_1 x1^(N-1) + ... + r_N
        ri = np.linalg.solve(A, b)
        ri = np.append(np.array([1]), ri) # add a 1 coefficient for the x0^n term
        r = np.append(r, np.array([ri]), axis=0)
        for i in np.arange(N+1):
            r[j-1, i-1] = (-1)**(N-i)*r[j-1, i-1]

        xi = np.roots(r[j, :])
        x = np.append(x, np.array([xi]), axis=0)
        x[j-1, :] = xi
        r[j-1, :] = ri

    return [x, r]

def q(r, x, j, i, k):

    if k == 0:
        return 1
    else:
        # Here the second index of r doesn't get a -1 because we added a 1 to the 0th spot of rj
        return r[j-1, k] - x[j-1, i-1]*q(r, x, j, i, (k-1))

def weights(j, i, r, x, matrix_array):
    """
    Compute integration weights for UDR integration

    :param j:
    :param i:
    :param r:
    :param x:
    :param matrix_array:
    :return float:
    """
    n = np.shape(matrix_array)[1]

    sigma = 0
    for k in np.arange(n): # k = 0 ... N-1
        sigma += (-1)**k * (-1)**(n-1)*matrix_array[j-1, n-k-1, -1] * q(r, x, j, i, k)

    product = 1
    for k in np.arange(1, n+1): # k = 1, k=/=i, up to n
        if k == i:
            term = 1
        else:
            term = x[j-1, i-1] - x[j-1, k-1]

        product = product * term

    return sigma/product

def UDR_evCalculation(j, y, l, xr, matrix_array, input_variables):
    """
    Approximate the lth moment of y(mu_1,...,xj,...,mu_N) with the UDR weighted sum.

    :param j:
    :param input_variables:
    :param y:
    :param l:
    :param xr:
    :param matrix_array:
    :return float:
    """
    if l == 0: # trivial case
        return 1

    x, r = xr
    sigma = 0
    n = np.shape(x)[1]
    node, key = input_variables[j - 1]
    for i in np.arange(1, n+1): # sum i = 1 ... n
        perturb = [node, key, x[j-1, i-1]]
        yval = y(perturb)
        sigma += weights(j, i, r, x, matrix_array) * (yval)**l

    return sigma

UDR_evCalculation.count = 0

def UDR_moment_approximation(exp, env, l, n):
    ## Compute the interpolation points
    error_params = get_error_parameters(exp)
    error_functions = get_error_functions(exp)
    fit_mean = simulate_with_error([0,0,0], exp, env)
    f2 = lambda x: simulate_with_error(x, exp, env) - fit_mean
    matrix_moments = compute_moment_matrices(error_params, n)
    x, r = compute_interpolation_points(matrix_moments)

    ## Make sure there wasn't any underflow errors etc
    xim = np.imag(x)
    xre = np.real(x)
    if np.any(np.imag(x) != 0):
        raise np.linalg.LinAlgError("Complex values found in interpolation points")
    x = np.real(x)

    ## Compute moments of the output distribution
    simulate_with_error.count = 0
    moment = UDR_moments(f2, l, error_params, error_functions, [x,r], matrix_moments)

    return moment

def UDRAnalysis(exp, env):
    """
    Function to fully do error analysis after the inner algorithm runs.

    :param exp:
    :param env:
    :return:
    """
    #Get parameters and compute interpolation points.
    error_params = get_error_parameters(exp)
    error_functions = get_error_functions(exp)
    fit_mean = simulate_with_error([0,0,0], exp, env)

    # We will compute the individual std of each component:

    variances = []
    f = lambda x: simulate_with_error(x, exp, env) - fit_mean

    j = 0
    for param in error_params:
        matrix_moments = compute_moment_matrices([param], 5)
        x, r = compute_interpolation_points(matrix_moments)

        xim = np.imag(x)
        xre = np.real(x)
        if np.any(np.imag(x) != 0):
            raise np.linalg.LinAlgError("Complex values found in interpolation points")

        x = xre
        variance = UDR_moments(f, 2, [param], [error_functions[j]], [x,r], matrix_moments)

        variances.append(variance)

    return variances

def multivariable_simulate(x, experiment, environment):

    at = experiment.getattributes()

    j = 0
    for node in experiment.node():
        i = 0
        y = experiment.node()[int(node)]['info'].at
        n_node = np.shape(y)[0]
        experiment.node()[int(node)]['info'].at = y*0
        experiment.node()[int(node)]['info'].at = y + x[j:n_node+j]
        j += n_node

        '''
        for optimal_val in y:
            mu, sigma = experiment.node()[int(node)]['info'].at_pdfs[i]
            upper = experiment.node()[int(node)]['info'].UPPER[i]
            lower = experiment.node()[int(node)]['info'].LOWER[i]

            val = x[j]
            real_val = val
            # Perturb is a triple with the error parameter key and the new value
            # co-ordinate xform
            #real_val = sigma*val + mu

            #real_val += optimal_val

            #if real_val > upper:
            #    real_val = upper
            #if real_val < lower:
            #    real_val = lower
            print(real_val)
            experiment.nodes()[int(node)]['info'].at[i] = real_val

            j += 1
        '''
    experiment.simulate(environment)
    At = experiment.nodes[experiment.measurement_nodes[0]]['output']

    fit = environment.waveform_temporal_overlap(At)

    # reset to optimal value
    experiment.setattributes(at)
    print(fit)
    return fit[0]

