#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Parent class for all node-types
"""

import networkx


class NodeType(object):
    """Parent class for node-type
    """

    __internal_var = 4

    def __init__(self, **kwargs):
        super().__init__()

        # TODO: should this be here? or is this bad practice
        if 'parameters' in kwargs:
            self._parameters = kwargs['parameters']
        else:
            self._parameters = []

        # TODO: this also feels a little hacky, but makes the front-end potentially easier to work with
        if 'parameters_from_name' in kwargs:
            if self.parameter_names:
                self._parameters = [None] * len(self.parameter_names)
                for (parameter_name, parameter_value) in kwargs['parameters_from_name'].items():
                    ind = self.parameter_names.index(parameter_name)
                    assert type(ind) == int
                    self._parameters[ind] = parameter_value
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
        if not (min(self._range_input_edges) <= number_input_edges <= max(self._range_input_edges)):
            raise TypeError("Current node, {}, has an unphysical number of inputs".format(self.__class__))
        if not (min(self._range_output_edges) <= number_output_edges <= max(self._range_output_edges)):
            raise TypeError("Current node, {}, has an unphysical number of outputs".format(self.__class__))
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

