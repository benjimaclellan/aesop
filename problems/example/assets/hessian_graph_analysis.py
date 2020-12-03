import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
import copy

# TODO: if both get_all_free_wheeling and get free_wheeling_<>_scores are used in code, refactor such that hessian is only computed once (it takes a long time)


def get_all_edge_scores(graph, as_log=True):
    """
    Returns dictionary with (key, value) = (node, terminal score), dictionary with (key, value) = (node, free_wheeling score) 
    """
    x, *_ = graph.extract_parameters_to_list()
    if len(x) == 0: # with bitstream -> photodiode, it's possible to have no unlocked parameters
        scores = {}
        for edge in graph.edges:
            scores[edge] = 0 if as_log else 1
        return scores, scores

    hess = normalize_hessian(graph.scaled_hess(x)) # if it crashes here, but NOT on the unboxed version, we have our problem isolated
    
    return free_wheeling_edge_scores(graph, hess, as_log=as_log)


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


def free_wheeling_edge_scores(graph, hess, p=1, as_log=False):
    """
    The free-wheeling score for node i, n_i is defined as:
    avg(p_j) for all parameters j in the node parameters (def of p_j found in free_wheeling_param_scores docstring)
    
    NOTE: only nodes with unlocked or temporarily locked parameters will be included in this scoring

    :param graph: graph on which to score nodes
    :returns: a dictionary of the node and its corresponding score
    """
    # only for testing...
    x, *_ = graph.extract_parameters_to_list()

    if len(x) != hess.shape[0]:
        raise ValueError(f'Hessian and parameter dimensionality mismatch. Parameter: {len(x)}, Hessian: {hess.shape}')

    param_scores = free_wheeling_param_scores(hess, p=p)

    return _extract_average_over_edge(graph, param_scores, as_log=as_log)

def free_wheeling_param_scores(hess, p=1):
    """
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


def _extract_average_over_edge(graph, vector_to_average, as_log=False):
    edge_scores = {}
    
    node_edge_list = graph.extract_attributes_to_list_experimental([], get_location_indices=False, \
                                                                   get_node_edge_index=True, exclude_locked=True)['node_edge_index']

    for edge in graph.edges:
        filter = np.array([node_edge == edge for node_edge in node_edge_list])
        filtered_scores = vector_to_average[filter]
        if len(filtered_scores) != 0:
            score = np.mean(filtered_scores)

            while type(score) == ArrayBox: # TODO: investigate. smh I end up with double-boxed floats... I don't know why
                score = score._value

            if as_log:
                score = np.log10(score)

            edge_scores[edge] = score
        else:
            edge_scores[edge] = 0 if as_log else 1
    
    return edge_scores
