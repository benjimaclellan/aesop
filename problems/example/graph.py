"""

"""

import networkx as nx
import copy
import matplotlib.pyplot as plt
from itertools import cycle

from utils.base_classes import Graph as GraphParent
from .assets.functions import power_, psd_


class Graph(GraphParent):
    """Parent class
    """

    __internal_var = None

    def __init__(self, nodes = dict(), edges = list(), propagate_on_edges = False):
        """
        """
        super().__init__()

        for node, model in nodes.items():
            self.add_node(node, **{'model': model, 'name': 'FEATURE NOT IMPLEMENTED', 'lock': False})

        # add the models to the edges if propagating on edges (fibers, paths etc)
        #TODO: this maybe could be improved - its a little hacky (but it works for now, so leave for now)

        for edge in edges:
            if len(edge) == 2:
                self.add_edge(edge[0], edge[1])
            elif len(edge) > 2:
                self.add_edge(edge[0], edge[1], **{'model': edge[2], 'name': "FUNCTIONALITY NOT IMPLEMENTED", 'lock': False, 'states':None})
            else:
                raise TypeError("Incorrect number of arguments in the {} edge connection tuple".format(edge))
        self._propagate_on_edges = propagate_on_edges

        self._propagation_order = None

        return


    def suc(self, node):
        """Return the successors of a node (nodes which follow the current one) as a list
        """
        return list(self.successors(node))

    def pre(self, node):
        """Return the predeccessors of a node (nodes which lead to the current one) as a list
        """
        return list(self.predecessors(node))

    def propagate(self, propagator):
        """
        """
        propagation_order = self.propagation_order
        for node in propagation_order:  # loop through nodes in the prescribed, physical order
            if not self.pre(node):  # check if current node has any incoming edges, if not, pass the node the null input propagator directly
                tmp_states = [copy.deepcopy(propagator.state)]  # nodes take a list of propagators as default, to account for multipath
            else:  # if we have incoming nodes to get the propagator from
                tmp_states = []  # initialize list to add all incoming propagators to
                for pre in self.pre(node):  # loop through incoming edges
                    if self._propagate_on_edges and 'model' in self.edges[(pre, node)]:  # this also simulates components stored on the edges, if there is a model on that edge
                        tmp_states += copy.deepcopy(self.edges[pre, node]['model'].propagate(self.nodes[pre]['states'], propagator, 1, 1))  # TODO: Check that models on edge are single spatial mode maybe
                    else:
                        tmp_states += copy.deepcopy(self.nodes[pre]['states'])

            # save the list of propagators at that node locations (deepcopy required throughout)
            self.nodes[node]['states'] = self.nodes[node]['model'].propagate(tmp_states, propagator, len(self.pre(node)), len(self.suc(node)))

        return self

    @property
    def propagation_order(self):
        """Returns the sorted order of nodes (based on which reverse walking the graph)
        """
        if nx.algorithms.recursive_simple_cycles(self):  # do not proceed if there is a cycle in the graph
            raise ValueError("There is a loop in the graph - current versions cannot simulate loops")

        _propagation_order = []  # initialize the list which defines the order nodes are visited
        node_set = set(self.nodes)  # use set type to compare which nodes have already been added
        for it in range(len(self.nodes)):  # maximum number of times the graph needs to be checked
            # for all nodes that haven't been added to _propagation_order, check which ones have no outgoing edges (terminal)
            terminal_nodes = [node for node in node_set if set(self.suc(node)).issubset(set(_propagation_order))]
            for terminal_node in terminal_nodes:
                _propagation_order = [terminal_node] + _propagation_order  # add these terminal nodes to the beginning of the order
            node_set = node_set - set(terminal_nodes)  # remove these terminal nodes from the our comparator set
        return _propagation_order

    @propagation_order.setter
    def propagation_order(self, propagation_order):
        self._propagation_order = propagation_order
        return

    def assert_number_of_edges(self):
        """Loops through all nodes and checks that the proper number of input/output edges are connected
        """
        for node in self.nodes:
            number_input_edges, number_output_edges = len(self.pre(node)), len(self.suc(node))
            self.nodes[node]['model'].assert_number_of_edges(number_input_edges, number_output_edges)
        if self._propagate_on_edges:
            for edge in self.edges:
                if 'model' in self.edges[edge]:
                    self.edges[edge]['model'].assert_number_of_edges(1, 1)  # by definition, edges have 1 in, 1 out
        return

    def sample_parameters(self, probability_dist='uniform', **kwargs):
        """ Samples new parameters for each node-type """
        for node in self.nodes:
            self.nodes[node]['model'].sample_parameters(probability_dist=probability_dist, **kwargs)

        if self._propagate_on_edges:
            for edge in self.edges:
                if 'model' in self.edges[edge]:
                    self.edges[tuple(edge)]['model'].sample_parameters(probability_dist=probability_dist, **kwargs)
        return

    def sample_parameters_to_list(self, probability_dist='uniform', **kwargs):
        """ Samples new parameters for each node-type """
        self.sample_parameters(probability_dist=probability_dist, **kwargs)
        parameters, node_edge_index, parameter_index, *_ = self.extract_parameters_to_list()
        return parameters, node_edge_index, parameter_index

    @staticmethod
    def extract_parameters(_node_edge, _model, _attribute, _model_attributes, *args):
        """ appends into _lst all unlocked attributes (from string _attribute) in _model
        args is always args[0] = node_edge_indices, and args[1] = parameter_indices
        """

        for i, attr in enumerate(getattr(_model, _attribute)):
            if not _model.parameter_locks[i]:
                _model_attributes[_attribute].append(attr)
                if args is not None:
                    args[0].append(_node_edge)
                    args[1].append(i)
        return


    def extract_attributes_to_list_experimental(self, attributes, get_location_indices=True):
        """ experimental: to extract model variables to a list, based on a list of variables names """

        # first generate multiple empty lists for each attribute of attributes
        model_attributes = {}
        for attribute in attributes:
            model_attributes[attribute] = []

        if get_location_indices:
            model_attributes['node_edge_index'], model_attributes['parameter_index'] = [], []  # these will help translate from a list of parameters/parameter info to a graph structure


        # we loop through all nodes and add the relevant info (bounds, etc.)
        for node in self.nodes:
            model = self.nodes[node]['model']
            if model.node_lock:
                continue
            for attribute in attributes:
                if get_location_indices:
                    self.extract_parameters(node, model, attribute, model_attributes,
                                            model_attributes['node_edge_index'],
                                            model_attributes['parameter_index'])
                else:
                    self.extract_parameters(node, model, attribute, model_attributes)


        # here, if considering edges too, we loop through and add them to each list (instead of a node hash, it is the edge tuple)
        if self._propagate_on_edges:
            for edge in self.edges:
                if 'model' in self.edges[edge]:
                    model = self.edges[edge]['model']
                    if model.node_lock:
                        continue
                    for attribute in attributes:
                        if get_location_indices:
                            self.extract_parameters(edge, model, attribute, model_attributes,
                                                    model_attributes['node_edge_index'],
                                                    model_attributes['parameter_index'])
                        else:
                            self.extract_parameters(edge, model, attribute, model_attributes)
        return model_attributes

    # TODO: we need to have this extract a dynamic selection (or always all) of the model characteristics (bounds, names, type, ...)
    def extract_parameters_to_list(self):
        """ Extracts the current parameters, bounds and information for re-distributing from the graph structure """
        # def extract_parameters(_node_edge, _model, _node_edge_index, _parameters_current, _parameter_index, _lower_bounds, _upper_bounds):
        #     parameter_details = zip(_model.parameters, _model.lower_bounds, _model.upper_bounds, _model.parameter_locks)
        #     for i, (parameter, low, up, lock) in enumerate(parameter_details):
        #         if not lock:
        #             _parameters_current.append(parameter)
        #             _node_edge_index.append(_node_edge)
        #             _parameter_index.append(i)
        #             _lower_bounds.append(low)
        #             _upper_bounds.append(up)
        #     return

        # parameters_current = []
        # node_edge_index, parameter_index = [], []  # these will help translate from a list of parameters to a graph structure
        # lower_bounds, upper_bounds = [], []
        #
        # # we loop through all nodes and add the relevant info (bounds, etc.)
        # for node in self.nodes:
        #     model = self.nodes[node]['model']
        #     if model.node_lock:
        #         continue
        #     extract_parameters(node, model, node_edge_index, parameters_current, parameter_index, lower_bounds, upper_bounds)
        #
        # # here, if considering edges too, we loop through and add them to each list (instead of a node hash, it is the edge tuple)
        # if self._propagate_on_edges:
        #     for edge in self.edges:
        #         if 'model' in self.edges[edge]:
        #             model = self.edges[edge]['model']
        #             if model.node_lock:
        #                 continue
        #             extract_parameters(edge, model, node_edge_index, parameters_current, parameter_index, lower_bounds, upper_bounds)

        attributes = ['parameters', 'lower_bounds', 'upper_bounds']
        model_attributes = self.extract_attributes_to_list_experimental(attributes, get_location_indices=True)
        parameters = model_attributes['parameters']
        node_edge_index = model_attributes['node_edge_index']
        parameter_index = model_attributes['parameter_index']
        lower_bounds = model_attributes['lower_bounds']
        upper_bounds = model_attributes['upper_bounds']
        return parameters, node_edge_index, parameter_index, lower_bounds, upper_bounds

    def distribute_parameters_from_list(self, parameters, node_edge_index, parameter_index):
        """ from the lists created in 'extract_parameters_to_list', we distribute these (or altered versions, like in scipy.optimize.minimize) back to the graph"""
        for i, (parameter, node_edge, parameter_ind) in enumerate(zip(parameters, node_edge_index, parameter_index)):
            if type(node_edge) is tuple:  # then we know this is an edge
                self.edges[node_edge]['model'].parameters[parameter_ind] = parameter
            else:  # anything else is a node
                self.nodes[node_edge]['model'].parameters[parameter_ind] = parameter
        return

    def inspect_parameters(self):
        """ Loops through all nodes & edge (if enabled) and prints information about the parameters """
        for node in self.nodes:
            self.nodes[node]['model'].inspect_parameters()
        if self._propagate_on_edges:
            for edge in self.edges:
                if 'model' in self.edges[edge]:
                    self.edges[edge]['model'].inspect_parameters()

    def inspect_state(self, propagator):
        """ we loop through all nodes and plot the optical state *after* the node"""
        fig, ax = plt.subplots(2, 1)
        linestyles = cycle(['-', '--', '-.', ':'])

        # please note that we do not include the edges here (most of the time I doubt we will use edges, but it may be useful in the future)
        for node in reversed(self.propagation_order):
            state = self.nodes[node]['states'][0]
            line = {'ls':next(linestyles), 'lw':3}
            ax[0].plot(propagator.t, power_(state), label=node, **line)
            ax[1].plot(propagator.f, psd_(state, propagator.dt, propagator.df), **line)

        ax[0].legend()
        plt.show()