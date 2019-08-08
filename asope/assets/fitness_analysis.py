import autograd.numpy as np
from autograd import grad
import scipy.integrate as integrate
from scipy.special import binom
from scipy.misc import factorial2
from copy import deepcopy
import sys
from autograd import elementwise_grad, jacobian
import pandas as pd
from tqdm import tqdm
import matplotlib.pylab as plt
import warnings

# %% Redundancy checks
def remove_redundancies(exp, env, verbose=False):
    """
    Check over an experiment for redundant nodes

    :param experiment:
    :param env:
    :param verbose:
    :return:
    """
    exp_redundancies = deepcopy(exp)
    exp_backup = deepcopy(exp)

    # Compute the fitness of the unmodified experiment, for comparison
    At = exp_redundancies.nodes[exp_redundancies.measurement_nodes[0]]['output']
    fit = env.fitness(At)
    valid_nodes = exp_redundancies.get_nonsplitters()

    for node in valid_nodes:
        exp_redundancies.remove_component(node)  # Drop a node from the experiment
        exp_redundancies.measurement_nodes = exp_redundancies.find_measurement_nodes()
        exp_redundancies.checkexperiment()
        exp_redundancies.make_path()
        exp_redundancies.check_path()
        exp_redundancies.inject_optical_field(env.At)
        # Simulate the optical table with dropped node
        exp_redundancies.simulate(env)
        At_mod = exp_redundancies.nodes[exp_redundancies.measurement_nodes[0]]['output']
        fit_mod = env.fitness(At_mod)
        if verbose:
            print(fit_mod, fit)
            print("Dropped node: {}. Fitness: {}".format(node, fit_mod))

        # Compare the fitness to the original optical table
        if fit_mod[0] >= fit[0]:  # and fit_mod[1] >= fit[1]:
            print("Dropping node")
            # If dropping the node didn't hurt the fitness, then drop it permanently
            fit = fit_mod
        else:
            exp_redundancies = deepcopy(exp_backup)

    return exp_redundancies



#
# #%% Monte Carlo
# def analysis_mc(at, exp, env, N=10**2, verbose=True):
#     """
#     """
#     if verbose:
#         print("Beginning Monte Carlo Simulation")
#
#     at_lst, node_lst, idx_lst, sigma_lst, mu_lst, at_names = exp.attributes_to_list(at)
#
#     at_mu, at_std = np.zeros_like(at_lst), np.zeros_like(at_lst)
#
#     sys.stdout.flush()
#     pbar = tqdm(total=100)
#     for k, (allele, node, idx, sigma, mu, name) in enumerate(zip(at_lst, node_lst, idx_lst, sigma_lst, mu_lst, at_names)):
#         fitnesses = np.zeros(N)
#         # print('\n\n')
#         # print((allele, node, idx, sigma, mu, name))
#
#         for i in range(N):
#             pbar.update(100 * (1 / N / len(at_lst)))
#             at_perturb = dict(at)
#             at_perturb[node][idx] += np.random.normal(mu, sigma)
#             exp.setattributes(at_perturb)
#             exp.simulate(env)
#             At = exp.nodes[exp.measurement_nodes[0]]['output']
#
#             # Evaluate the fitness and store it in the array
#             fitnesses[i] = env.fitness(At)[0]
#
#         at_mu[k] = np.mean(fitnesses)
#         at_std[k] = np.std(fitnesses)
#     pbar.close()
#
#     if verbose:
#         comp_labels = []
#         at_labels = []
#         for node, key in get_error_parameters(exp):
#             # Get the component labels and construct a string for the dataframe
#             node, key = int(node), int(key)
#             try:
#                 title = exp.nodes()[node]['title']
#                 label = exp.nodes()[node]['info'].AT_NAME[key]
#                 comp_labels.append(title)
#                 at_labels.append(label)
#             except AttributeError or TypeError:
#                 print("Error getting component and attribute labels")
#
#         results = pd.DataFrame(comp_labels, columns=["Component"])
#         results = results.assign(Attribute=at_labels)
#         results = results.assign(Output_Deviation=at_std)
#         print(results)
#         print("Monte Carlo finished\n\n")
#
#     # Reset the exp to optimal parameters
#     exp.setattributes(at)
#     return at_mu, at_std





#
# def get_error_parameters(experiment):
#     """
#     Iterate through the list of nodes in the experiment, and append their error keyvals into an array.
#     The components of the array are tuples of the form [node, key] which has the dictionary key 'node'
#     indicating which node and 'key' indicating which error parameter.
#
#     :param experiment:
#     :return:
#     """
#     error_params = np.zeros((0, 2))
#     for node in experiment.nodes():
#         i = 0
#         node_at = experiment.node()[node]['info'].at
#         for val in node_at:
#             tuple = np.array([[node, i]])
#             i += 1
#             error_params = np.append(error_params, tuple, axis=0)
#
#     return error_params
#
# def get_error_functions(experiment):
#     """
#     Iterature through the list of nodes in the experiment, and append their error pdf keyvals into an array.
#
#     :param experiment:
#     :return:
#     """
#     error_functions = np.zeros(0)
#
#     for node in experiment.nodes():
#         '''
#         node_ep = experiment.nodes()[node]['info'].ERROR_PDFS
#         for key in node_ep:
#             fx = node_ep[key]
#             error_functions = np.append(error_functions, fx)
#         '''
#
#         node_pdfs = (experiment.node()[node]['info'].MU, experiment.node()[node]['info'].SIGMA)
#         for fx in node_pdfs:
#             error_functions = np.append(error_functions, fx)
#
#     return error_functions

#
# def simulate_with_error(perturb, experiment, environment):
#     simulate_with_error.count += 1
#     """
#     Given an error perturbation, return the fitness.
#
#     Param should be a triple of the form [node, key, val] where node and key are dictionary keys identifying
#     which error parameter is being perturbed, and val is the value of the perturbation.
#
#     Currently, this code will only work for one fitness parameter.
#
#     :param perturb:
#     :param experiment:
#     :param environment:
#     :return float:
#     """
#     # Perturb is a triple with the error parameter key and the new value
#     node, index, val = perturb
#
#     if index - int(index) != 0:
#         raise ValueError("All attribute indices should be integers!")
#     index = int(index)
#
#     mu, sigma = experiment.node()[int(node)]['info'].MU[index], experiment.node()[int(node)]['info'].SIGMA[index]
#     upper = experiment.node()[int(node)]['info'].UPPER[index]
#     lower = experiment.node()[int(node)]['info'].LOWER[index]
#
#     # get the current value
#     optimal_val = experiment.nodes()[int(node)]['info'].at[index]
#
#     # Perturb is a triple with the error parameter key and the new value
#     # co-ordinate xform
#     real_val = sigma*val + mu
#
#     real_val += optimal_val
#
#     if real_val > upper:
#         real_val = upper
#     if real_val < lower:
#         real_val = lower
#
#     experiment.nodes()[int(node)]['info'].at[index] = real_val
#
#     experiment.simulate(environment)
#     At = experiment.nodes[experiment.measurement_nodes[0]]['output']
#     fit = environment.fitness(At)
#
#     # reset to optimal value
#     experiment.node()[int(node)]['info'].at[index] = optimal_val
#
#     return fit[0]
#
# simulate_with_error.count = 0

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

#%% Univariate dimension reduction


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

# UDR_evCalculation.count = 0

def UDR_moment_approximation(exp, env, l, n):
    ## Compute the interpolation points
    error_params = get_error_parameters(exp)
    error_functions = get_error_functions(exp)
    fit_mean = simulate_with_error([0,0,0], exp, env)
    f2 = lambda x: simulate_with_error(x, exp, env) - fit_mean
    matrix_moments = compute_moment_matrices(error_params, n)
    x, r = compute_interpolation_points(matrix_moments)

    ## Make sure there wasn't any underflow errors etc
#    xim = np.imag(x)
#    xre = np.real(x)
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

#        xim = np.imag(x)
        xre = np.real(x)
        if np.any(np.imag(x) != 0):
            raise np.linalg.LinAlgError("Complex values found in interpolation points")

        x = xre
        variance = UDR_moments(f, 2, [param], [error_functions[j]], [x,r], matrix_moments)

        variances.append(variance)

    return variances



def analysis_udr(at, exp, env, verbose=True):
    """
    Function to fully do error analysis after the inner algorithm runs.

    :param exp:
    :param env:
    :return:
    """
    if verbose:
        print("Beginning univariate dimension reduction")

    exp.setattributes(at)
    exp.simulate(env)
    At = exp.nodes[exp.measurement_nodes[0]]['output']
    fit = env.fitness(At)
    
    stds = np.sqrt(UDRAnalysis(exp, env))

    if verbose: 
        comp_labels = []
        at_labels = []
        for node, key in get_error_parameters(exp):
            # Get the component labels and construct a string for the dataframe
            node, key = int(node), int(key)
            # try:
            title = exp.nodes()[node]['title']
            label = exp.nodes()[node]['info'].AT_NAME[key]
            comp_labels.append(title)
            at_labels.append(label)
            # except AttributeError or TypeError:
            #     print("Error getting component and attribute labels")

        results = pd.DataFrame(comp_labels, columns=["Component"])
        results = results.assign(Attribute = at_labels)
        results = results.assign(Output_Deviation = stds)
        print(results)
        print("Univariate dimension reduction finished\n\n")
    return stds


# #%% Landscape Hessian Analaysis
# def multivariable_simulate(x, experiment, environment):
#     """
#     Function which preforms an experiment with a given input parameter vector x, and then computes and returns the
#     fitness of the result. This allows you to run autograd on it.
#     """
#
#     # at = experiment.getattributes()
#     # Get the optimal parameters mu
#     # mu = experiment.attributes_to_list(at)
#     mu, _, _, sigma_lst, _, _ = experiment.attributes_to_list(at)
#     # Get the deviation of the parameters
#     # sigma_list = experiment.get_sigma_list()
#
#     # Co-ordinate transform to unitless co-ordinates
#     x = (x - np.array(mu))/np.array(sigma_lst)
#
#     # Iterate through x and set the parameters of the experiment
#     j = 0
#     for node in experiment.node():
#         y = experiment.node()[int(node)]['info'].at
#         n_node = np.shape(y)[0]
#         experiment.node()[int(node)]['info'].at = y*0
#         experiment.node()[int(node)]['info'].at = y + x[j:n_node+j]
#         j += n_node
#
#     # Simulate and compute fitness
#     experiment.simulate(environment)
#     At = experiment.nodes[experiment.measurement_nodes[0]]['output']
#
#     fit = environment.fitness(At)
#
#     # reset to optimal value
#     experiment.setattributes(at)
#     return fit[0]

def autograd_hessian(fun, argnum = 0):
    '''
    Compute the hessian by computing the transpose of the jacobian of the gradient.

    :param fun:
    :param argnum:
    :return:
    '''

    def sum_latter_dims(x):
        return np.sum(x.reshape(x.shape[0], -1), 1)

    def sum_grad_output(*args, **kwargs):
        return sum_latter_dims(elementwise_grad(fun)(*args, **kwargs))

    return jacobian(sum_grad_output, argnum)


# def analysis_lha(at, exp, env, symmetry_tol=1e-5, verbose=True):
#
#     print('Beginning landscape hessian analysis')
#
#     f = lambda x: multivariable_simulate(x, exp, env)
#     # Compute the Hessian of the fitness function (as a function of x)
#     Hf = autograd_hessian(f)
#
#     # Construct a vector of the mean value, and a vector of the standard deviations.
#     # muv, sigma_list, basis, at_name = [], [], [], []
#     # j,k = (0, 0)
#
#     at_lst, node_lst, idx_lst, sigma_lst, mu_lst, at_names = exp.attributes_to_list(at)
#
#     # for node in exp.nodes():
#     #     for name in exp.nodes[node]['info'].AT_NAME:
#     #         at_name.append('{}:{}'.format(node,name))
#     #     for q in at[node]:
#     #         muv.append(q)
#     #         basis.append(node)
#     #         j += 1
#     #     for mu, sigma in exp.nodes[node]['info'].at_pdfs:
#     #         sigma_list.append(sigma)
#     #         k += 1
#
#     # muv, sigma_list, basis = np.array(muv), np.array(sigma_list), np.array(basis)
#     H0 = Hf(np.array(mu_lst))/2
#
#     sym_dif = H0 - H0.T
# #    if np.amax(sym_dif) > symmetry_tol:
# #        raise ValueError("Max asymmetry is large " + str(np.amax(sym_dif)))
#
#     # Compute eigenstuff of the matrix, and sort them by eigenvalue magnitude
#     eigen_items = np.linalg.eig(H0)
#     eigensort_inds = np.argsort(eigen_items[0])
#     eigenvalues = eigen_items[0][eigensort_inds]
#     eigenvectors = eigen_items[1][:,eigensort_inds]
#
#     basis_names = ([i for i in range(0, len(at_names))], at_names)
#
#     return H0, eigenvalues, eigenvectors, basis_names


def lha_analysis_wrapper(x_opt, func, mu_lst, sigma_lst):
    # Compute the Hessian of the fitness function (as a function of x)
    Hf = autograd_hessian(func)
    H0 = Hf(0 * np.array(mu_lst)) / 2

    symmetry_tol = 1e-5
    sym_dif = H0 - H0.T
    if np.amax(sym_dif) > symmetry_tol:
       warnings.warn("Max asymmetry is large {}".format(np.amax(sym_dif)))

    # Compute eigenstuff of the matrix, and sort them by eigenvalue magnitude
    eigen_items = np.linalg.eig(H0)
    eigensort_inds = np.argsort(eigen_items[0])
    eigenvalues, eigenvectors = eigen_items[0][eigensort_inds], eigen_items[1][:, eigensort_inds]

    return np.diag(H0), H0, eigenvalues, eigenvectors




def udr_analysis_wrapper(x_opt, func, mu_lst, sigma_lst):
    return






def mc_analysis_wrapper(x_opt, func, mu_lst, sigma_lst, N=10**2):
    """
    """
    analysis_mu, analysis_std = np.zeros_like(x_opt), np.zeros_like(x_opt)
    sys.stdout.flush()
    with tqdm(total=100) as pbar:
        for k, (xi, mu, sigma) in enumerate(zip(x_opt, mu_lst, sigma_lst)):
            fitnesses = np.zeros(N)
            for i in range(N):
                pbar.update(100 * (1 / N / len(x_opt)))
                x_perturb = list(x_opt)
                x_perturb[k] += np.random.normal(mu, sigma)
                fitnesses[i] = func(x_perturb)
            analysis_mu[k] = np.mean(fitnesses)
            analysis_std[k] = np.std(fitnesses)
    sys.stdout.flush()
    print('\n')
    return analysis_std, analysis_mu