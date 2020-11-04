import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox


def lock_free_wheeling_params(graph):
    pass
    # TODO: add easy locking/unlocking mechanism (analogous to assigning param values from list)


def unlock_non_free_wheeling_params(graph):
    pass


def get_all_free_wheeling_params(graph, grad_thresh=1e-5, hess_thresh=1e-5, test_point_num=2):
    x, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()

    grad_assessment = np.abs(graph.grad(x)) < grad_thresh
    hessian_assessment = np.abs(graph.scaled_hess(x)) < hess_thresh
    hessian_part1 = np.all(hessian_assessment, axis=0)
    hessian_part2 = np.all(hessian_assessment, axis=1)

    is_free_wheeling = np.bitwise_and(grad_assessment, np.bitwise_and(hessian_part1, hessian_part2))
    for _ in range(test_point_num):
        p = graph.sample_parameters_to_list()
        grad_assessment = np.abs(graph.grad(p)) < grad_thresh
        hessian_assessment = np.abs(graph.scaled_hess(p)) < hess_thresh
        hessian_part1 = np.all(hessian_assessment, axis=0)
        hessian_part2 = np.all(hessian_assessment, axis=1)

        is_free_wheeling = np.bitwise_and(is_free_wheeling, np.bitwise_and(grad_assessment, np.bitwise_and(hessian_part1, hessian_part2)))

    return np.array(node_edge_index)[is_free_wheeling], np.array(parameter_index)[is_free_wheeling]


def normalize_hessian(hessian, evaluator):
    """
    How to normalize is a big unknown. There needs to be something,
    otherwise the choice of evaluator will potentially affect the Hessian
    - Do we normalize with respect to a standard graph?
    - Do we need to normalize with respect to number of parameters?
    - Can we define something on the hessian itself?

    My gut feeling is that we need to normalize the same way
    across all graphs of an optimization but dunno
    """
    return hessian


def free_wheeling_node_score(graph, node):
    pass

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

