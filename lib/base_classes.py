"""Parent classes
"""

import networkx
import uuid
import config.config as configuration

class NodeType(object):
    """Parent class for node-type
    """

    protected = False

    def __init__(self, **kwargs):
        super().__init__()

        self._parameters = self.default_parameters

        # sets the parameters from a list passed in at construction. must be a list with length of number_of_parameters, in the correct order
        if 'parameters' in kwargs:  # sets parameters based only on the order they are given in the list
            self._parameters = kwargs['parameters']
        # set parameters by name, can selectively set parameters (don't need to set all at construction, inherits the defaults if not passed in)
        elif 'parameters_from_name' in kwargs:  # sets parameters based on a dictionary of name/value pairs
            for (parameter_name, parameter_value) in kwargs['parameters_from_name'].items():
                self.set_parameter_from_name(parameter_name, parameter_value)
        else:
            self._parameters = self.default_parameters


        self.parameters_uuid = [uuid.uuid4() for i in range(self.number_of_parameters)] # unique id for each parameter for various purposes


        self.transform = None  # this is a variable which will store a visual representation of the transformation
        self.noise_model = None
        self._all_params = None

    @property
    def parameters(self):
        return self._parameters

    def set_parameters(self, parameters):
        self._parameters = parameters
        self.set_parameters_as_attr()
        self.update_noise_model() # TODO: maybe change this to be conditional?

    @parameters.setter
    def parameters(self, parameters):
        self.set_parameters(parameters)


    def assert_number_of_edges(self, number_input_edges, number_output_edges):
        """  """
        if not (min(self._range_input_edges) <= number_input_edges <= max(self._range_input_edges)):
            raise TypeError("Current node, {}, has an unphysical number of inputs ({}). Correct range is {}-{}".format(self.__class__, number_input_edges,
                                                                                                                       min(self._range_input_edges), max(self._range_input_edges)))
        if not (min(self._range_output_edges) <= number_output_edges <= max(self._range_output_edges)):
            raise TypeError("Current node, {}, has an unphysical number of outputs ({})Correct range is {}-{}".format(self.__class__, number_output_edges,
                                                                                                                       self._range_output_edges, self._range_input_edges))
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
        """ Prints out information about the parameters for this model """
        print('Current node {}, parameters {}'.format(self, self.parameters))
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
    
    def update_noise_model(self):
        pass


class Evaluator(object):
    """Parent class
    """

    def __init__(self):
        super().__init__()
        return



class Graph(networkx.MultiDiGraph):
    """Parent class
    """

    def __init__(self, **attr):
        super().__init__(**attr)
        return


class EvolutionOperators(object):
    """Parent class
    """

    def __init__(self, verbose=False, **attr):
        super().__init__(**attr)
        self.verbose = verbose
        self.source_models = set(configuration.NODE_TYPES_ALL['SourceModel'].values())
        self.sink_models = set(configuration.NODE_TYPES_ALL['SinkModel'].values())
        self.edge_models = set(configuration.NODE_TYPES_ALL['SinglePath'].values())
        self.node_models = set(configuration.NODE_TYPES_ALL['MultiPath'].values())
    
    def apply_evolution(self, graph, location):
        """
        Applies evolution operator at location,
        where location is a node, edge, or interface.
        """
        raise NotImplementedError('Base class evolution operator is an interface, has no implementation itself')

    def possible_evo_locations(self, graph):
        """
        Returns a set of all possible locations on which the evolution may be applied
        Locations can be nodes, edges, or interfaces, depending on the evolution operator
        """
        raise NotImplementedError('Base class evolution operator is an interface, has no implementation itself')


class TerminalNode(NodeType):
    """Parent class
    """

    protected = True

    def __init__(self, **attr):
        super().__init__(**attr)
        return
