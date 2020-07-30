
import scipy.optimize
import cma

"""
"""

def parameters_optimize(graph, x0=None, method='L-BFGS', options):
    if method == 'L-BFGS':


    elif method == 'CMA'
        _, node_edge_index, parameter_index, lower_bounds, upper_bounds = graph.extract_parameters_to_list()
        es = cma.CMAEvolutionStrategy(parameters_initial, 0.999,
                                      {'verb_disp': 1, 'maxfevals': 1000, 'bounds': [lower_bounds, upper_bounds]})
        es.optimize(graph.func)
        res = es.result
        parameters = res.xbest
        
    elif method == 'ADAM':


    elif method == 'GA':


    else:
        raise ModuleNotFoundError('This is not a defined minimization method')

# TODO: Not a todo, but a note that we should ALWAYS by default use a minimizing function
def parameters_minimize(graph, parameters_initial):
    """  """
    # TODO: sp.optimize.minimize can use the Hessian matrix - this may simplify our lives
    _, node_edge_index, parameter_index, lower_bounds, upper_bounds = graph.extract_parameters_to_list()


    print("Starting parameter minimization with scipy.optimize builtin tools")
    res = scipy.optimize.minimize(graph.func, parameters_initial, method='L-BFGS-B',
                                  bounds=list(zip(lower_bounds, upper_bounds)),
                                  options={'disp':True, 'maxiter':1000},
                                  jac=graph.grad)
    graph.distribute_parameters_from_list(res.x, node_edge_index, parameter_index)

    return res.x, res


# def function_to_optimize(parameters, graph, evaluator, propagator, node_edge_index, parameter_index):
#     graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)  # distribute the current parameter values (stored in a list, as this is the datatype of sp.optimize) to the graph & each node/edge
#     graph.propagate(propagator)
#     score = evaluator.evaluate_graph(graph, propagator)
#     return score
