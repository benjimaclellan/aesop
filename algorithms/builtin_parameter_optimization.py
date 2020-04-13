import scipy.optimize


"""
BY DEFINITION, LET'S ALWAYS MINIMIZE - THAT IS HOW SCIPY WORKS, SO LETS KEEP IT THAT WAY
"""



def minimize_parameters(graph, evaluator, propagator, info=False):
    """  """
    # TODO: sp.optimize.minimize can use the Hessian matrix - this may simplify our lives
    parameters_initial, node_index, parameter_index, lower_bounds, upper_bounds = initialize_minimize_parameters(graph)

    def func(parameters):
        return function_to_optimize(parameters, graph, evaluator, propagator, node_index, parameter_index)

    # options for bounded: L-BFGS-B, TNC, SLSQP and trust-constr
    # if adding hess or jac, only option is trust-constr
    print("Starting parameter minimization with scipy.optimize builtin tools")
    res = scipy.optimize.minimize(func, parameters_initial, method='trust-constr',
                                  bounds=list(zip(lower_bounds, upper_bounds)),
                                  options={'disp':True})

    if info:
        return res.x
    else:
        return res.x, res

def initialize_minimize_parameters(graph):
    """  """
    parameters_initial = []
    node_index, parameter_index = [], []  # these will help translate from a list of parameters to a graph structure
    lower_bounds, upper_bounds = [], []
    for node in graph.nodes:
        model = graph.nodes[node]['model']

        if model.node_lock:
            continue

        parameter_details = zip(model.parameters, model.lower_bounds, model.upper_bounds, model.parameter_locks)

        for i, (parameter, low, up, lock) in enumerate(parameter_details):
            if not lock:
                parameters_initial.append(parameter)
                node_index.append(node)
                parameter_index.append(i)
                lower_bounds.append(low)
                upper_bounds.append(up)

    return parameters_initial, node_index, parameter_index, lower_bounds, upper_bounds


def distribute_parameters_to_graph(parameters, graph, node_index, parameter_index):
    for i, (parameter, node, parameter_ind) in enumerate(zip(parameters, node_index, parameter_index)):
        graph.nodes[node]['model'].parameters[parameter_ind] = parameter
    return

def function_to_optimize(parameters, graph, evaluator, propagator, node_index, parameter_index):
    distribute_parameters_to_graph(parameters, graph, node_index, parameter_index)
    graph.propagate(propagator)
    score = evaluator.evaluate_graph(graph, propagator)
    return score