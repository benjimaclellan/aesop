import numpy as np

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


def simulate_component_noise(experiment, environment, input_At, N_samples, record=True):
    """
    Runs a monte carlo simulation of the component noise. Returns a (N_samples, m) array, where m is the number
    of parameters returned by environment.fitness

    :param experiment:
    :param environment:
    :param input_At:
    :param record:
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
    optical_fields = np.zeros((N_samples, length))
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
