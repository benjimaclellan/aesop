import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox

# TODO: if both get_all_free_wheeling and get free_wheeling_<>_scores are used in code, refactor such that hessian is only computed once (it takes a long time)


def get_all_node_scores(graph, as_log=True):
    """
    Returns dictionary with (key, value) = (node, terminal score), dictionary with (key, value) = (node, free_wheeling score) 
    """
    x, *_ = graph.extract_parameters_to_list()
    hess = normalize_hessian(graph.scaled_hess(x))

    return terminal_node_scores(graph, as_log=as_log), free_wheeling_node_scores(graph, hess, as_log=as_log)


def normalize_hessian(hessian):
    """
    How to normalize is a big unknown. There needs to be something,
    otherwise the choice of evaluator will potentially affect the Hessian
    - Do we normalize with respect to a standard graph?
    - Do we need to normalize with respect to number of parameters?
    - Can we define something on the hessian itself?

    My gut feeling is that we need to normalize the same way
    across all graphs of an optimization but dunno
    """
    return hessian / np.sum(hessian) * hessian.shape[0]**2


def normalize_gradient(grad):
    return grad / np.sum(grad) * grad.shape[0]


def free_wheeling_node_scores(graph, hess, p=1, as_log=False):
    """
    The free-wheeling score for node i, n_i is defined as:
    avg(p_j) for all parameters j in the node parameters (def of p_j found in free_wheeling_param_scores docstring)
    
    NOTE: only nodes with unlocked or temporarily locked parameters will be included in this scoring

    :param graph: graph on which to score nodes
    :returns: a dictionary of the node and its corresponding score
    """
    param_scores = free_wheeling_param_scores(hess, p=p)
    return _extract_average_over_node(graph, param_scores, as_log=as_log)


def terminal_node_scores(graph, as_log=False):
    x, _, _, lower_bounds, upper_bounds = graph.extract_parameters_to_list()

    x, lower_bounds, upper_bounds = np.array(x), np.array(lower_bounds), np.array(upper_bounds)
    uncertainty_scaling = graph.extract_attributes_to_list_experimental(['parameter_imprecisions'],
                                                                        get_location_indices=False,
                                                                        exclude_locked=True)['parameter_imprecisions']
    per_param_dist = np.minimum(x - lower_bounds, upper_bounds - x) / np.array(uncertainty_scaling) 

    return _extract_average_over_node(graph, per_param_dist, as_log=as_log)


# def terminal_node_scores(graph, grad):
#     return _extract_average_over_node(graph, np.abs(grad))


def free_wheeling_param_scores(hess, p=1):
    """
    TODO: take into account the gradient as well (this does not)

    We define the free-wheeling score of parameter i, p_i as:

    p_i = ( sum(abs(Hij)^p) + sum(abs(Hji)^p) )^(1/p) / 2sum(j)
    Where the sums are for all j in [0, <total graph parameter num>)
    and p is a given parameter, with p = 1 by default

    The function returns an numpy array [p_0, p_1, ..., p_{<graph param num> - 1}]
    
    NOTE: only unlocked or temporarily locked parameters will be included in this scoring

    :params graph: graph on which to compute parameter scores
    """
    hess_p = np.abs(hess)**p
    return (np.sum(hess_p, axis=0)**(1/p) + np.sum(hess_p, axis=1)**(1/p)) / (2 * hess.shape[0])


def _extract_average_over_node(graph, vector_to_average, as_log=False):
    node_scores = {}
    _, node_edge_index, *_ = graph.extract_parameters_to_list()

    for node in graph.nodes:
        filtered_scores = vector_to_average[np.array(node_edge_index) == node]
        if len(filtered_scores) != 0:
            score = np.mean(filtered_scores)

            while type(score) == ArrayBox: # TODO: investigate. smh I end up with double-boxed floats... I don't know why
                score = score._value

            if as_log:
                score = np.log10(score)

            node_scores[node] = score
    
    return node_scores










# ----------------------------------- Useful? -----------------------------------------

def free_wheeling_param(graph, i, grad_thresh=1e-5, hess_thresh=1e-5, test_point_num=2):
    """
    Operates on a single parameter
    Returns True if the parameter is deemed free-wheeling, False otherwise

    :pre-condition: Hessian must be properly normalized, exclude_locked_params = True in the grad/hess initializations

    TODO: verify that parameters do in fact need to be restored.. I suspect not
    """
    # 1. Save graph parameters such that they can be restored
    x, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
    
    # 2. test free-wheeling 
    params = x
    for _ in range(test_point_num):
        if not _free_wheeling_at_point(graph.grad(params), graph.scaled_hess(params), i, grad_thresh, hess_thresh):
            return False

        params = graph.sample_parameters_to_list()

    return True


def _free_wheeling_at_point(gradient, hessian, param_index, grad_thresh, hess_thresh):
    if type(gradient) == ArrayBox:
        gradient = gradient._value
    if type(hessian) == ArrayBox:
        hessian = hessian._value

    second_orders = np.concatenate((hessian[param_index, :], hessian[:, param_index]))
    is_free_wheeling = (np.abs(gradient[param_index]) < grad_thresh) and all(np.abs(second_orders) < hess_thresh)

    return is_free_wheeling

def lock_free_wheeling_params(graph):
    pass
    # TODO: add easy locking/unlocking mechanism (analogous to assigning param values from list)


def unlock_non_free_wheeling_params(graph):
    pass


def get_all_free_wheeling_params(graph, grad_thresh=1e-7, hess_thresh=1e-5, test_point_num=1):
    """
    Returns node_edge_index (1D numpy array), parameter_index (1D numpy array) of all free wheeling params
    """
    x, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()

    grad_assessment = np.abs(graph.grad(x)) < grad_thresh
    hessian_assessment = np.abs(graph.scaled_hess(x)) < hess_thresh
    hessian_part1 = np.all(hessian_assessment, axis=0)
    hessian_part2 = np.all(hessian_assessment, axis=1)

    is_free_wheeling = np.bitwise_and(grad_assessment, np.bitwise_and(hessian_part1, hessian_part2))
    for _ in range(test_point_num - 1):
        p = graph.sample_parameters_to_list()
        grad_assessment = np.abs(graph.grad(p)) < grad_thresh
        hessian_assessment = np.abs(graph.scaled_hess(p)) < hess_thresh
        hessian_part1 = np.all(hessian_assessment, axis=0)
        hessian_part2 = np.all(hessian_assessment, axis=1)

        is_free_wheeling = np.bitwise_and(is_free_wheeling, np.bitwise_and(grad_assessment, np.bitwise_and(hessian_part1, hessian_part2)))

    return np.array(node_edge_index)[is_free_wheeling], np.array(parameter_index)[is_free_wheeling]


def get_all_free_wheeling_nodes_params(graph, grad_thresh=1e-5, hess_thresh=1e-5, test_point_num=1):
    """
    Returns set of free wheeling nodes, np array of node edge indices of all free-wheeling params, np array of parameter indices of all free-wheeling params
    
    Note: this is a binary scheme: either you're free-wheeling or you're not
    """
    free_wheeling = set()
    node_edge_index, parameter_index = get_all_free_wheeling_params(graph, grad_thresh=grad_thresh, hess_thresh=hess_thresh, test_point_num=test_point_num)

    for node in graph.nodes:
        node_model = graph.nodes[node]['model']
        total_unlocked_node_num = node_model.number_of_parameters - np.count_nonzero(np.array(node_model.parameter_locks))
        if (np.count_nonzero(node_edge_index == node) == total_unlocked_node_num) and (total_unlocked_node_num != 0):
            free_wheeling.add(node)
    
    return free_wheeling, node_edge_index, parameter_index
