"""Parent classes
"""

import networkx
import warnings

from config.config import np

class NodeType(object):
    """Parent class for node-type
    """

    __internal_var = 4

    def __init__(self, **kwargs):
        super().__init__()

        self.parameter = self.default_parameters

        # TODO: this also feels a little hacky, but makes the front-end potentially easier to work with
        # TODO: should this be here? or is this bad practice
        if 'parameters' in kwargs:  # sets parameters based only on the order they are given in the list
            self._parameters = kwargs['parameters']

        elif 'parameters_from_name' in kwargs:  # sets parameters based on a dictionary of name/value pairs
            assert len(self.parameter_names) == self.number_of_parameters
            self._parameters = [None] * self.number_of_parameters
            for (parameter_name, parameter_value) in kwargs['parameters_from_name'].items():
                self.set_parameter_from_name(parameter_name, parameter_value)

        else:
            self._parameters = self.default_parameters


        return

    @property
    def parameters(self):
        return self._parameters

    def set_parameters(self, parameters):
        self._parameters = parameters

    @parameters.setter
    def parameters(self, parameters):
        self.set_parameters(parameters)


    def assert_number_of_edges(self, number_input_edges, number_output_edges):
        """  """
        if not (min(self._range_input_edges) <= number_input_edges <= max(self._range_input_edges)):
            raise TypeError("Current node, {}, has an unphysical number of inputs".format(self.__class__))
        if not (min(self._range_output_edges) <= number_output_edges <= max(self._range_output_edges)):
            raise TypeError("Current node, {}, has an unphysical number of outputs".format(self.__class__))
        return

    def get_parameter_from_name(self, parameter_name):
        """ Returns the parameter value from a given parameter name """
        return self._parameters[self.parameter_names.index(parameter_name)]


    def set_parameter_from_name(self, parameter_name, parameter_value):
        """  """
        ind = self.parameter_names.index(parameter_name)
        assert type(ind) == int
        self._parameters[ind] = parameter_value
        return

    def set_parameters_as_attr(self):
        """  """
        for name, parameter in zip(self.parameter_names, self.parameters):
            setattr(self, '_' + name, parameter)
        return

    def inspect_parameters(self):
        """  """
        for ind in range(self.number_of_parameters):
            try:
                print('Parameter name: {}, parameter value: {}\n\tupper bound: {}, lower bound: {}\n\tdata type: {}, step size: {}\n\tunit {}'.format(
                    self.parameter_names[ind], self.parameters[ind],
                    self.upper_bounds[ind], self.lower_bounds[ind],
                    self.data_types[ind], self.step_sizes[ind],
                    self.parameter_units[ind]
                ))
            except:
                raise Warning('Settings for this parameter are not all added to the class. This could cause errors.'
                              'Please check that all node-type attributes are correctly set.')
        return


    def sample_parameters(self, probability_dist = 'uniform', **kwargs):
        """ Samples the new parameters from a given distribution and parameter bounds """

        if self.node_lock:
            return

        parameter_details = zip(self.parameters, self.lower_bounds, self.upper_bounds,
                                self.data_types, self.step_sizes, self.parameter_locks)
        parameters = []
        for ind, (parameter_old, low, up, data_type, step, lock) in enumerate(parameter_details):
            if lock:
                parameters.append(parameter_old)
                continue
            if probability_dist == 'uniform':
                parameter = uniform_sample(low=low, up=up, step=step, data_type=data_type)
            elif probability_dist == 'triangle':
                interval_width = kwargs['interval_width'] if hasattr(kwargs, 'interval_width') else 0.05
                parameter = triangle_sample(parameter=parameter_old, low=low, up=up,
                                            step=step, data_type=data_type, interval_width=interval_width)
            else:
                warnings.warn("This is not a valid sampling function, reverting to uniform")
                parameter = uniform_sample(low=low, up=up, step=step, data_type=data_type)
            parameters.append(parameter)
        self.parameters = parameters
        return

class Evaluator(object):
    """Parent class
    """

    __internal_var = None

    def __init__(self):
        super().__init__()
        return



class Graph(networkx.DiGraph):
    """Parent class
    """

    __internal_var = None

    def __init__(self, **attr):
        super().__init__(**attr)
        return


class EvolutionOperators(object):
    """Parent class
    """

    __internal_var = None

    def __init__(self, **attr):
        super().__init__(**attr)
        return


## %

def uniform_sample(low=0.0, up=1.0, step=None, data_type='int'):
    if data_type == 'float':
        if step is None:
            parameter = np.random.uniform(low, up)
        else:
            parameter = round(np.random.uniform(low, up) / step) * step
    elif data_type == 'int':
        if step is None:
            parameter = np.random.randint(low, up)
        else:
            parameter = np.round(np.random.randint(low / step, up / step)) * step
    else:
        raise ValueError('Unknown datatype in the current parameter')
    return parameter

def triangle_sample(parameter=None, low=0.0, up=1.0, step=None, data_type='int', interval_width = 0.05):
    """
    Experimental mutation operator, using triangular probability distribution
    This is od-hoc - one foreseeable issue is that the probability of drawing up | low is always 0
    """

    if parameter is None:
        raise RuntimeError("Current parameter must be passed to apply a Gaussian mutation")

    radius = interval_width * (up - low) / 2
    left = parameter - radius if parameter - radius > low else low
    right = parameter + radius if parameter + radius < up else up
    parameter_old = float(parameter)
    if data_type == 'float':
        if step is None:
            parameter = np.random.triangular(left, parameter, right)
        else:
            parameter = round(np.random.triangular(left, parameter, right) / step) * step
    elif data_type == 'int':
        parameter = int(np.random.triangular(left, parameter, right))
    else:
        raise ValueError('Unknown datatype for this parameter')

    print('parameter_old {} | parameter {} | radius {} | left {} | right {}'.format(parameter_old, parameter, radius, left, right))

    return parameter