import autograd.numpy as np
from tqdm import tqdm
import sys
import warnings

#from analysis.lha import autograd_hessian
#from analysis.udr import compute_interpolation_points, compute_moment_matrices, udr_moments
from lha import autograd_hessian
from udr import compute_interpolation_points, compute_moment_matrices, udr_moments


def udr_analysis_wrapper(x_opt, func, mu_lst, sigma_lst, x_r_matrix_moments=None, m_num_moments=5):
    """

    """
    #Get parameters and compute interpolation points.
    fit_mean = func(x_opt)
    updated_func = lambda x: func(x) - fit_mean

    # We will compute the individual std of each component:
    stds = []
    N_parameters = len(x_opt)

    ##TODO: add an initialization for UDR so the matrices, etc can be computed once only
    # if x_r_matrix_moments = None:  # if this was not initialized, run it now
    for parameter_i, (mu, sigma) in enumerate(zip(mu_lst, sigma_lst)):
        matrix_moments = compute_moment_matrices(N_parameters, mu, sigma, m_num_moments)
        x, r = compute_interpolation_points(matrix_moments)
        
        xre = np.real(x)
        if np.any(np.imag(x) != 0):
            raise np.linalg.LinAlgError("Complex values found in interpolation points")
        x = xre
    # else:
    #     (x, r, matrix_moments) = x_r_matrix_moments



    for parameter_i, (mu, sigma) in enumerate(zip(mu_lst, sigma_lst)):
        variance = udr_moments(updated_func, 2, x_opt, parameter_i, N_parameters, mu, sigma, x, r, matrix_moments)
        stds.append(np.sqrt(variance))

    return stds,


def lha_analysis_wrapper(x_opt, func, mu_lst, sigma_lst, Hf=None):
    """

    """
    # Compute the Hessian of the fitness function (as a function of x), or pass in from initialization
    if Hf == None:
        Hf = autograd_hessian(func)

    H0 = Hf(np.array(x_opt) - np.array(mu_lst)) / np.array(sigma_lst) / 2

    symmetry_tol = 1e-5
    sym_dif = H0 - H0.T
    if np.amax(sym_dif) > symmetry_tol:
       warnings.warn("Max asymmetry is large {}".format(np.amax(sym_dif)))

    # Compute eigenstuff of the matrix, and sort them by eigenvalue magnitude

    eigen_items = np.linalg.eig(H0)
    eigensort_inds = np.argsort(eigen_items[0])
    eigenvalues, eigenvectors = eigen_items[0][eigensort_inds], eigen_items[1][:, eigensort_inds]

    return np.diag(H0), H0, eigenvalues, eigenvectors






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
    return analysis_std,