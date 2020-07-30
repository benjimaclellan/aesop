
import scipy.optimize

"""
"""

# TODO: Not a todo, but a note that we should ALWAYS by default use a minimizing function
def parameters_minimize(graph, propagator, evaluator, info=False):
    """  """
    # TODO: sp.optimize.minimize can use the Hessian matrix - this may simplify our lives
    parameters_initial, node_edge_index, parameter_index, lower_bounds, upper_bounds = graph.extract_parameters_to_list()

    def func(parameters):
        return function_to_optimize(parameters, graph, evaluator, propagator, node_edge_index, parameter_index)

    # options for bounded: L-BFGS-B, TNC, SLSQP and trust-constr
    # if adding hess or jac, only option is trust-constr
    print("Starting parameter minimization with scipy.optimize builtin tools")
    res = scipy.optimize.minimize(func, parameters_initial, method='L-BFGS-B',
                                  bounds=list(zip(lower_bounds, upper_bounds)),
                                  options={'disp':True, 'maxiter':1000})

    graph.distribute_parameters_from_list(res.x, node_edge_index, parameter_index)
    if info:
        return res.x
    else:
        return res.x, res


def function_to_optimize(parameters, graph, evaluator, propagator, node_edge_index, parameter_index):
    graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)  # distribute the current parameter values (stored in a list, as this is the datatype of sp.optimize) to the graph & each node/edge
    graph.propagate(propagator)
    score = evaluator.evaluate_graph(graph, propagator)
    return score
