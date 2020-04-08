#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import networkx as nx

from utils.parents import Graph as GraphParent


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

        #TODO: this maybe could be improved - its a little hacky (but it works for now, so leave for now)
        for edge in edges:
            if len(edge) == 2:
                self.add_edge(edge[0], edge[1])
            elif len(edge) > 2:
                self.add_edge(edge[0], edge[1], **{'model': edge[2], 'name': "FUNCTIONALITY NOT IMPLEMENTED", 'lock': False})
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
                tmp_propagator = [propagator]  # nodes take a list of propagators as default, to account for multipath
            else:  # if we have incoming nodes to get the propagator from
                tmp_propagator = []  # initialize list to add all incoming propagators to
                for pre in self.pre(node):  # loop through incoming edges
                    if self._propagate_on_edges and hasattr(self.edges[pre, node], 'model'):  # this also simulates components stored on the edges, if there is a model on that edge
                        tmp_propagator += self.edges[pre, node]['model'].propagate(self.nodes[pre]['propagator'])
                    else:
                        tmp_propagator += self.nodes[pre]['propagator']

            # save the list of propagators at that node locations
            self.nodes[node]['propagator'] = self.nodes[node]['model'].propagate(tmp_propagator)

        return

    @property
    def propagation_order(self):
        """Returns the sorted order of nodes (based on which reverse walking the graph)
        """
        if nx.algorithms.recursive_simple_cycles(self):  # do not proceed if there is a cycle in the graph
            raise ValueError("There is a loop in the graph")

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
                if hasattr(self.edges[edge[0], edge[1]], 'model'):
                    self.edges[edge[0], edge[1]]['model'].assert_number_of_edges(1, 1)  # by definition, edges have 1 in, 1 out
        return