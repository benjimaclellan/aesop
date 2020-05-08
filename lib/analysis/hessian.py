import autograd.numpy as np
from autograd import elementwise_grad, jacobian
import warnings


def hessian(func, argnum = 0):
    '''
    Compute the hessian by computing the transpose of the jacobian of the gradient.
    '''

    def sum_latter_dims(x):
        return np.sum(x.reshape(x.shape[0], -1), 1)

    def sum_grad_output(*args, **kwargs):
        return sum_latter_dims(elementwise_grad(func)(*args, **kwargs))

    return jacobian(sum_grad_output, argnum)

def get_hessian(graph, propagator, evaluator, exclude_locked=True):
    func = function_wrapper(graph, propagator, evaluator, exclude_locked=exclude_locked)
    hessian_function = hessian(func)
    return hessian_function

def get_scaled_hessian(graph, propagator, evaluator, exclude_locked=True):
    hessian_function = get_hessian(graph, propagator, evaluator, exclude_locked=exclude_locked)

    uncertainty_scaling = graph.extract_attributes_to_list_experimental(['parameter_imprecisions'],
                                                                        get_location_indices=False,
                                                                        exclude_locked=exclude_locked)['parameter_imprecisions']

    scale_factor = np.dot(np.array([uncertainty_scaling]).T, np.array([uncertainty_scaling]))

    scaled_hessian_function = lambda parameters: hessian_function(parameters) * scale_factor
    return scaled_hessian_function


def lha_analysis(parameters, func, uncertainty_scaling, hessian_function = None):
    """

    """
    # Compute the Hessian of the fitness function (as a function of x), or pass in from initialization
    if hessian_function is None:
        hessian_function = hessian(func)

    H0 = hessian_function(np.array(parameters)) / 2.0

    scale_mat = np.dot(np.array([uncertainty_scaling]).T, np.array([uncertainty_scaling]))
    H0 *= scale_mat

    # the hessian should be symetric, check if this is the case
    SYMMETRY_TOL = 1e-5
    sym_dif = H0 - H0.T
    if np.amax(sym_dif) > SYMMETRY_TOL:
        warnings.warn("Max asymmetry is large {}".format(np.amax(sym_dif)))

    # Compute eigenstuff of the matrix, and sort them by eigenvalue magnitude

    eigen_items = np.linalg.eig(H0)
    eigensort_inds = np.argsort(eigen_items[0])
    eigenvalues, eigenvectors = eigen_items[0][eigensort_inds], eigen_items[1][:, eigensort_inds]

    return np.diag(H0), H0, eigenvalues, eigenvectors

def function_wrapper(graph, propagator, evaluator, exclude_locked = False):
    """ returns a function handle that accepts only parameters and returns the score. used to initialize the hessian analysis """

    def _function(_parameters, _graph, _propagator, _evaluator, _node_edge_index, _parameter_index):
        _graph.distribute_parameters_from_list(_parameters, _node_edge_index, _parameter_index)
        _graph.propagate(_propagator)
        score = _evaluator.evaluate_graph(_graph, _propagator)
        return score

    info = graph.extract_attributes_to_list_experimental([], get_location_indices=True,
                                                         exclude_locked=exclude_locked)

    func = lambda parameters: _function(parameters, graph, propagator, evaluator,
                                        info['node_edge_index'], info['parameter_index'])
    return func