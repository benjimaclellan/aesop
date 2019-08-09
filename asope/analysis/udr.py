import autograd.numpy as np
import scipy.integrate as integrate
from scipy.special import binom

'''
j indexes the random variable i.e. j in [1,N]
i indexes the moment i.e. i in [1,l]

'''

def normal_pdf(x, mu=0, sigma=1):
    """
    Probability distribution function of normal distribution.
    """
    scale = 1/(np.sqrt(2*np.pi*sigma**2))
    return scale * np.exp(-1*(x-mu)**2/(2*sigma**2))


def compute_interpolation_points(matrix_array):
    """
    Invert the moment matrix to get polynomial coefficients and find the roots, which are the interpolation points
    for the UDR integral approximation.
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


def evintegrand(val, error_distribution, m):
    """
        Returns the integrand for the expected value function. Meant to be called by expectationValueM.
        At the moment, we only use a normal distribution for all error parameters. Changing this will require some significant re-engingeering of the code flow
    """
    return val**m * error_distribution(val)


def expectationValueM(error_distribution, m):
    """
    Compute the expectation value of Y^m (mu_i,...,x_j,...,mu_N) for a given error parameter x_j and function func.
    """
    expected_value = integrate.quad(evintegrand, -np.inf, np.inf, args=(error_distribution, m))
    return expected_value[0]

'''
Functions from here down are used for UDR calculations. The conventions are as follows;
j in [1,...,N] indexes the input random variables
i in [1,...,n] indexes the interpolation points

for python arrays indexed starting from 0, we must always call the array element i-1 or j-1

UDR = univariate dimensional reduction. see Rahman and Xu (2004)
'''

def S(i, j, func, matrix_array, x_opt, parameter_i, N_parameters, mu, sigma, x, r):
    """

    Computes Sij, coefficients required for UDR. Recursive function that recurses on j.
    Sij = Sum(k=0...i) iCk Skj-1 E(Y^(i-k)(mu1,...,Xj,...,muN)
    """

    if j<1:
        # j should range from N to 1, never less
        raise ValueError("j should never be less then 1")

    if j==1:
        # The base case, Si1 = E[Y^i(X_1, mu_2, ... , mu_N)]
        base = udr_evcalc(j, func, i, x_opt, parameter_i, N_parameters, mu, sigma, x, r, matrix_array)
        return base

    else:
        parameter_sigma = 0.0
        for k in np.arange(i+1): # Sum k=0 up to i
            # Compute S recursively
            S_val = S(k, j - 1, func, matrix_array, x_opt, parameter_i, N_parameters, mu, sigma, x, r)
            parameter_sigma += binom(i, k) * udr_evcalc(j, func, i - k, x_opt, parameter_i, N_parameters, mu, sigma, x, r, matrix_array) * S_val
        return parameter_sigma


def compute_moment_matrices(N_parameters, mu, sigma, n):
    """
    To find the interpolation points we need to construct a matrix

    |mu_(j, n-1), ... , (-1)^(n-1) mu_(j, 0) |
    |mu_(j, n)  , ... , (-1)^(n-1) mu_(j, 1) |
    |   ...     , ... ,         ...          |
    |mu(j, 2n-2), ... , (-1)^(n-1) mu_(j,n-1)|

    this function will return a 3d matrix, where the jth element is the above 2d matrix
    """
    ## We want to solve Ax = b but need to find A and b
    matrix_array = np.zeros((0, n, n+1))
    moment_matrix = np.zeros([n, n+1])

    for j in np.arange(1, N_parameters + 1): # j=1,..,N
        error_distribution = normal_pdf  #here we do a coordinate transform instead

        # Here we construct an n x n+1 matrix, which we will split into a n dimensional column vector and a n x n dimensional matrix M (the moment matrix)
        for i in np.arange(1, n+2):  # i = 1,...,n+1
            for k in np.arange(1, n+1):  # k = 1,...,n
                mu_ik = ((-1)**i)*expectationValueM(error_distribution, n-i+k)
                if i == 1:  # The 0th column should be positive as I am going to slice it off as b
                    mu_ik = -1*mu_ik
                moment_matrix[k-1, i-1] = mu_ik

        matrix_array = np.append(matrix_array, np.array([moment_matrix]), axis=0)

    return matrix_array


def udr_evcalc(j, func, l, x_opt, parameter_i, N_parameters, mu, sigma, x, r, matrix_array):
    """
    Approximate the lth moment of func(mu_1,...,xj,...,mu_N) with the UDR weighted sum.
    """
    if l == 0:  # trivial case
        return 1

    parameter_sigma = 0
    n = np.shape(x)[1]
    for i in np.arange(1, n+1): # sum i = 1 ... n
        val_sample_point = x[j - 1, i - 1]

        x_perturb = list(x_opt)
        x_perturb[parameter_i] += val_sample_point * sigma + mu  # this is if we do a coord transform after

        yval = func(x_perturb)
        parameter_sigma += weights(j, i, r, x, matrix_array) * yval**l

    return parameter_sigma

def udr_moments(func, l, x_opt, parameter_i, N_parameters, mu, sigma, x, r, matrix_array):

    """
    Compute the lth moment of Y(error_params) using univariate dimension reduction.
    See 'A univariate dimension-reduction method for multi-dimensional integration in stochastic mechanics'
    by Rahman and Xu (2004) for an exposition of the technique.
    """
    moment = 0.0

    for i in np.arange(l+1): # sum i = 0 to l
        sval = S(i, N_parameters, func, matrix_array, x_opt, parameter_i, N_parameters, mu, sigma, x, r)  # Compute S^i_N
        moment += binom(l, i)*sval*(-(N_parameters-1)*0)**(l-i)

    return moment
