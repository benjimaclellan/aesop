import numpy as np
from assets.graph_manipulation import get_nonsplitters
from copy import deepcopy

def update_error_attributes(experiment):
    """
    Goes through each component in the experiment and gives them new sampled error parameters

    :param experiment:
    :return:
    """
    exp = experiment
    for node in exp.node():
        # Check that there is an error model to update
        try:
            if exp.node()[node]['info'].N_EPARAMETERS != 0:
                component = exp.node()[node]['info']
                #error_model() will return an array of appropriately sampled error attributes
                #update_error_attributes() will use the array to save the properties to the component
                new_at = component.error_model()

                component.update_error_attributes(new_at)
        except AttributeError:
            print("Exception occurred updating error models - is there a component without an implemented error model?")


def simulate_component_noise(experiment, environment, input_At, N_samples):
    """
    Runs a monte carlo simulation of the component noise. Returns a (N_samples, m) array, where m is the number
    of parameters returned by environment.fitness

    :param experiment:
    :param environment:
    :param input_At:
    :param N_samples:
    :return:
    """
    #rename variables
    At = input_At
    env = environment
    exp = experiment
    m = np.shape(env.fitness(At))[0]
    #create empty array with the correct shape
    fitnesses = np.zeros((N_samples, m))
    length = np.shape(At)[0]
    optical_fields = np.zeros((N_samples, length), dtype=np.complex)
    for i in np.arange(N_samples):
        update_error_attributes(exp)
        # Simulate the experiment with the sampled error profile
        exp.simulate(env)
        At = exp.nodes[exp.measurement_nodes[0]]['output']
        optical_fields[i] = At[:, 0]
        # Evaluate the fitness and store it in the array
        fit = env.fitness(At)
        fitnesses[i] = fit

    return fitnesses, optical_fields

def drop_node(experiment, node):
    """
        Remove a specific single node

    :param experiment:
    :param node:
    """
#    print('remove_one_node')

    pre = experiment.pre(node)
    suc = experiment.suc(node)

    if len(pre) == 1:
        experiment.remove_edge(pre[0], node)
    if len(suc) == 1:
        experiment.remove_edge(node, suc[0])
    if len(pre) == 1 and len(suc) == 1:
        experiment.add_edge(pre[0], suc[0])
    experiment.remove_node(node)
    return experiment

def remove_redundancies(experiment, env, verbose=False):
    """
    Check over an experiment for redundant nodes

    :param experiment:
    :param env:
    :param verbose:
    :return:
    """
    exp = experiment
    # Compute the fitness of the unmodified experiment, for comparison
    At = exp.nodes[exp.measurement_nodes[0]]['output']
    fit = env.fitness(At)
    valid_nodes = get_nonsplitters(exp)

    for node in valid_nodes[:]:
        if verbose:
            print("Dropping node: " + str(node))
        exp_bak = deepcopy(exp)
        exp_mod = drop_node(exp_bak, node) # Drop a node from the experiment
        exp_mod.measurement_nodes = exp_mod.find_measurement_nodes()
        exp_mod.checkexperiment()
        exp_mod.make_path()
        exp_mod.check_path()
        exp_mod.inject_optical_field(env.At)
        # Simulate the optical table with dropped node
        exp_mod.simulate(env)
        At_mod = exp_mod.nodes[exp_mod.measurement_nodes[0]]['output']
        fit_mod = env.fitness(At_mod)
        if verbose:
            print("Obtained fitness of " + str(fit_mod))
            print("-_-_-_-")
        # Compare the fitness to the original optical table
        if fit_mod[0] >= fit[0]: #and fit_mod[1] >= fit[1]:
            print("Dropping node")
            # If dropping the node didn't hurt the fitness, then drop it permanently
            exp = exp_mod
            fit = fit_mod
            At = At_mod

    return exp
