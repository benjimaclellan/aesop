
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

import autograd.numpy as np
from autograd import hessian
from autograd import elementwise_grad, jacobian
import warnings


def hessian_local(func, argnum = 0):
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


def lha_analysis(hessian_function, parameters):
    """

    """

    H0 = hessian_function(np.array(parameters))

    # the hessian should be symetric, check if this is the case
    SYMMETRY_TOL = 1e-5
    sym_dif = H0 - H0.T
    if np.amax(sym_dif) > SYMMETRY_TOL:
        warnings.warn("Max asymmetry is large {}".format(np.amax(sym_dif)))

    # Compute eigenstuff of the matrix, and sort them by eigenvalue magnitude

    eigen_items = np.linalg.eig(H0)
    eigensort_inds = np.argsort(np.abs(eigen_items[0]))
    eigenvalues, eigenvectors = eigen_items[0][eigensort_inds], eigen_items[1][:, eigensort_inds]
    return np.diag(H0), H0, eigenvalues, eigenvectors


def plot_eigenvectors(parameter_names, eigenvectors, eigenvalues):
    fig, ax = plt.subplots(1,1)

    norm = mpl.colors.Normalize(vmin=min(eigenvalues), vmax=max(eigenvalues))
    cmap = cm.viridis



    xticks = list(range(len(parameter_names)))
    for i, (evec, eval) in enumerate(zip(eigenvectors, eigenvalues)):
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        markerline, stemlines, baseline = ax.stem(xticks, evec, use_line_collection=True, markerfmt='o')
        markerline.set_markerfacecolor(m.to_rgba(eval))
    ax.set(xticks=xticks)
    ax.set_xticklabels(parameter_names, rotation=45, ha='center')
    return ax

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