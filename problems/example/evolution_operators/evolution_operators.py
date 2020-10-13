import numpy as np
import random
import networkx as nx


import config.config as configuration
from lib.decorators import register_evolution_operators
from lib.base_classes import EvolutionOperators as EvolutionOperators
from algorithms.speciation import Speciation
import random
import matplotlib.pyplot as plt


@register_evolution_operators
class AddNode(EvolutionOperators):
    """
    Evolution operator: add one single-path node
    """
    potential_node_types = set(['SinglePath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        """ Applies the specific evolution on the graph argument
        """

        # find all potential model types (all single-path nodes) to choose one from
        potential_nodes = []
        for node_type in self.potential_node_types:
            potential_nodes += list(configuration.NODE_TYPES_ALL[node_type].values())
        model = random.sample(potential_nodes, 1)[0]()

        # choose a random edge to insert the node at
        edge = random.sample(graph.edges, 1)[0]

        # find an appropriate label for the new node (no duplicates in the graph)
        node_label = min(set(range(min(graph.nodes), max(graph.nodes)+2)) - set(graph.nodes))
        graph.add_node(node_label, **{'model': model, 'name': model.__class__.__name__, 'lock': False})
        if graph.speciation_descriptor is not None and graph.speciation_descriptor['name'] == 'photoNEAT':
            historical_marker = Speciation.next_historical_marker()
            graph.speciation_descriptor['node to marker'][node_label] = historical_marker
            graph.speciation_descriptor['marker to node'][historical_marker] = node_label

        # update graph connections
        graph.add_edge(edge[0], node_label)
        graph.add_edge(node_label, edge[1])
        graph.remove_edge(edge[0], edge[1])

        if verbose:
            print('Evolution operator: AddNode | Adding node {} with model {} on edge {}'.format(node_label, model, edge))
        return graph

    def verify_evolution(self, graph):
        """ Checks if the specific evolution on the graph is possible, returns bool

        All logic for going through graph structures and what node specifics there are, and returning a Yes/No of whether this
        evolution operator can be applied to this graph
        """
        # TODO check graph here
        return True # just for now


@register_evolution_operators
class RemoveNode(EvolutionOperators):

    potential_node_types = set(['SinglePath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        """ Applies the specific evolution on the graph argument
        """

        # find all possible nodes to remove (all single-path nodes in graph)
        potential_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in self.potential_node_types:
                potential_nodes.append(node)
        node_to_remove = random.sample(potential_nodes, 1)[0]
        if verbose:
            print('Evolution operator: RemoveNode | Removing node {} with model {}'.format(node_to_remove, graph.nodes[node_to_remove]['model']))


        # update graph connections
        print(f'pre: {graph.pre(node_to_remove)}')
        print(f'suc: {graph.suc(node_to_remove)}')
        try:
            graph.add_edge(graph.pre(node_to_remove)[0], graph.suc(node_to_remove)[0])
        except IndexError as e:
            print(e)
            graph.draw(legend=True)
            plt.show()
        graph.remove_edge(graph.pre(node_to_remove)[0], node_to_remove)
        graph.remove_edge(node_to_remove, graph.suc(node_to_remove)[0])

        graph.remove_node(node_to_remove)
        if graph.speciation_descriptor is not None and graph.speciation_descriptor['name'] == 'photoNEAT':
            marker = graph.speciation_descriptor['node to marker'].pop(node_to_remove)
            graph.speciation_descriptor['marker to node'].pop(marker)

        return graph

    def verify_evolution(self, graph):
        """ Checks if the specific evolution on the graph is possible, returns bool

        All logic for going through graph structures and what node specifics there are, and returning a Yes/No of whether this
        evolution operator can be applied to this graph
        """
        potential_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in self.potential_node_types:
                potential_nodes.append(node)
        if len(potential_nodes) > 0:
            return True
        else:
            return False



@register_evolution_operators
class SwapNode(EvolutionOperators):

    potential_node_types = set(['SinglePath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        """ Applies the specific evolution on the graph argument
        """

        # find all nodes which could be swapped and randomly select one
        swappable_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in self.potential_node_types:
                swappable_nodes.append(node)
        node_to_swap = random.sample(swappable_nodes, 1)[0]

        potential_new_nodes = []
        for node_type in self.potential_node_types:
            potential_new_nodes += list(configuration.NODE_TYPES_ALL[node_type].values())

        # choose new model and swap out the model on that node
        new_model = random.sample(potential_new_nodes, 1)[0]()
        graph.nodes[node_to_swap]['model'] = new_model
        graph.nodes[node_to_swap]['name'] = new_model.__class__.__name__

        if verbose:
            print('Evolution operator: SwapNode | Swapping node {} from model {} to model {}'.format(node_to_swap, graph.nodes[node_to_swap]['model'], new_model))
        return graph

    def verify_evolution(self, graph):
        """ Checks if the specific evolution on the graph is possible, returns bool

        All logic for going through graph structures and what node specifics there are, and returning a Yes/No of whether this
        evolution operator can be applied to this graph
        """
        potential_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in self.potential_node_types:
                potential_nodes.append(node)
        if len(potential_nodes) > 0:
            return True
        else:
            return False


@register_evolution_operators
class AddInterferometer(EvolutionOperators):

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        """ Applies the specific evolution on the graph argument
        """

        # create instance of two new multi-path nodes (just 2x2 couplers for now)
        splitter1 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()
        splitter2 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()

        # choose two separate edges to insert the new multi-path nodes
        edges = random.sample(graph.edges, 2)

        # add new splitters into graph
        node_labels = set(range(min(graph.nodes), max(graph.nodes) + 4)) - set(graph.nodes)
        label1 = min(node_labels)
        label2 = min(node_labels - set([label1]))

        graph.add_node(label1, **{'model': splitter1, 'name': splitter1.__class__.__name__, 'lock': False})
        graph.add_node(label2, **{'model': splitter2, 'name': splitter2.__class__.__name__, 'lock': False})
        if graph.speciation_descriptor is not None and graph.speciation_descriptor['name'] == 'photoNEAT':
            for node in [label1, label2]:
                historical_marker = Speciation.next_historical_marker()
                graph.speciation_descriptor['node to marker'][node] = historical_marker
                graph.speciation_descriptor['marker to node'][historical_marker] = node


        # we need to put the new splitters in the proper order to avoid loops (follow the propagation order)
        if graph.propagation_order.index(edges[0][0]) <= graph.propagation_order.index(edges[1][0]):
            edge1, edge2 = edges[0], edges[1]
        else:
            edge1, edge2 = edges[1], edges[0]

        # fix connections in graph
        graph.remove_edge(edge1[0], edge1[1])
        graph.remove_edge(edge2[0], edge2[1])

        graph.add_edge(edge1[0], label1)
        graph.add_edge(label1, edge1[1])
        graph.add_edge(label2, edge2[1])
        graph.add_edge(edge2[0], label2)

        graph.add_edge(label1, label2)

        if verbose:
            print('Evolution operator: AddInterferometer | Splitters added at edges {} and {}'.format(edge1, edge2))

        return graph

    def verify_evolution(self, graph):
        """ Checks if the specific evolution on the graph is possible, returns bool

        All logic for going through graph structures and what node specifics there are, and returning a Yes/No of whether this
        evolution operator can be applied to this graph
        """
        potential_node_types = set(['MultiPath'])
        multipath_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in potential_node_types:
                multipath_nodes.append(node)
        if len(multipath_nodes) > 20 or len(graph.edges) < 3:
            return False
        else:
            return True


@register_evolution_operators
class RemoveOneInterferometerPath(EvolutionOperators):
    """
    Collapses an interferometer structure to a single path (if possible)
    Note: using the complex AddInterferometer can cause this function for removing paths to fail on occasion
    """
    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        """ Applies the specific evolution on the graph argument
        """

        # conver graph to a simple (undirected, non-hyper, simple graph) to find cycles
        tmp_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            w = 1.0
            if tmp_graph.has_edge(u, v):
                tmp_graph[u][v]['weight'] += w
            else:
                tmp_graph.add_edge(u, v, weight=w)

        cycles = nx.cycle_basis(tmp_graph)

        # collapsed cycles are when beamsplitters link to each other on both edges (this info is lost when converted to simple undirected graph)
        collapsed_cycles = [edge for edge in tmp_graph.edges if tmp_graph.get_edge_data(edge[0], edge[1])['weight'] > 1]
        cycles += collapsed_cycles

        # we choose a random cycle that we will consider
        cycle = random.sample(cycles, 1)[0] # chooses random cycle to remove part of

        # use the old propagation order to ensure physicality when placing new nodes
        propagation_order = graph.propagation_order

        # find the source/sink node of the cycle by comparing to the propagation order (first in order is the source, last is sink)
        propagation_index = [propagation_order.index(node) for node in cycle]
        source_node = cycle[np.argmin(propagation_index)]
        sink_node = cycle[np.argmax(propagation_index)]

        # choose one branch to remain in the graph, we will delete the others
        branch_to_keep = random.sample(graph.suc(source_node), 1)[0]
        if branch_to_keep == sink_node:
            branch_to_keep = None

        # check what nodes/edges to keep and remove
        paths_from_source_to_sink = list(nx.all_simple_paths(graph, source_node, sink_node))
        paths_to_remove = [path for path in paths_from_source_to_sink if branch_to_keep not in path]
        nodes_to_remove = set([item for sublist in paths_to_remove for item in sublist])
        paths_to_keep = [path for path in paths_from_source_to_sink if branch_to_keep in path]
        nodes_to_keep = set([item for sublist in paths_to_keep for item in sublist]) - set([source_node, sink_node])

        # connect graph back together deal with the possibilities of removing a whole loop
        if not nodes_to_keep:
            graph.add_edge(graph.pre(source_node)[0], graph.suc(sink_node)[0])
        else:
            tmp = list(nodes_to_keep)
            propagation_index = [propagation_order.index(node) for node in tmp]
            new_source_node = tmp[np.argmin(propagation_index)]
            new_sink_node = tmp[np.argmax(propagation_index)]

            graph.add_edge(graph.pre(source_node)[0], new_source_node)
            graph.add_edge(new_sink_node, graph.suc(sink_node)[0])

        # remove the nodes that should be deleted
        graph.remove_nodes_from(nodes_to_remove)
        if graph.speciation_descriptor is not None and graph.speciation_descriptor['name'] == 'photoNEAT':
            for node in nodes_to_remove:
                marker = graph.speciation_descriptor['node to marker'].pop(node)
                graph.speciation_descriptor['marker to node'].pop(marker)            

        if verbose:
            str_info = 'Source node {} | Sink node {} | Branch to keep {} | Nodes to delete {}'.format(source_node, sink_node, branch_to_keep, nodes_to_remove)
            print('Evolution operator: RemoveOneInterferometerPath | {}'.format(str_info))

        return graph


    def verify_evolution(self, graph):
        """ Checks if the specific evolution on the graph is possible, returns bool

        All logic for going through graph structures and what node specifics there are, and returning a Yes/No of whether this
        evolution operator can be applied to this graph
        """

        # check to see if there is a cycle which can be collapses
        tmp_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            w = 1.0
            if tmp_graph.has_edge(u, v):
                tmp_graph[u][v]['weight'] += w
            else:
                tmp_graph.add_edge(u, v, weight=w)
        cycles = nx.cycle_basis(tmp_graph)
        collapsed_cycles = [edge for edge in tmp_graph.edges if tmp_graph.get_edge_data(edge[0], edge[1])['weight'] > 1]
        cycles += collapsed_cycles

        if len(cycles) > 0:
            return True
        else:
            return False


# @register_evolution_operators
class AddInterferometerSimple(EvolutionOperators):

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        """ Applies the specific evolution on the graph argument
        """

        # two new multipath splitters (just 2x2 couplers for now)
        splitter1 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()
        splitter2 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()

        # insert a random single-path node
        potential_node_types = set(['SinglePath'])
        potential_nodes = []
        for node_type in potential_node_types:
            potential_nodes += list(configuration.NODE_TYPES_ALL[node_type].values())
        new_nonsplitter = random.sample(potential_nodes, 1)[0]()

        # edge to add interferometer in
        edge = random.sample(graph.edges, 1)[0]

        node_labels = set(range(min(graph.nodes), max(graph.nodes) + 4)) - set(graph.nodes)
        label1 = min(node_labels)
        label2 = min(node_labels - set([label1]))
        label3 = min(node_labels - set([label1]) - set([label2]))

        graph.add_node(label1, **{'model': splitter1, 'name': splitter1.__class__.__name__, 'lock': False})
        graph.add_node(label2, **{'model': splitter2, 'name': splitter2.__class__.__name__, 'lock': False})
        graph.add_node(label3, **{'model': new_nonsplitter, 'name': new_nonsplitter.__class__.__name__, 'lock': False})
        if graph.speciation_descriptor is not None and graph.speciation_descriptor['name'] == 'photoNEAT':
            for node in [label1, label2, label3]:
                historical_marker = Speciation.next_historical_marker()
                graph.speciation_descriptor['node to marker'][node] = historical_marker
                graph.speciation_descriptor['marker to node'][historical_marker] = node

        graph.remove_edge(edge[0], edge[1])
        graph.add_edge(edge[0], label1)
        graph.add_edge(label2, edge[1])

        graph.add_edge(label1, label2)
        graph.add_edge(label1, label3)
        graph.add_edge(label3, label2)

        if verbose:
            print('Evolution operator: AddInterferometerSimple | Edge to insert at {} with node {}'.format(edge, new_nonsplitter))

        return graph

    def verify_evolution(self, graph):
        """ Checks if the specific evolution on the graph is possible, returns bool

        All logic for going through graph structures and what node specifics there are, and returning a Yes/No of whether this
        evolution operator can be applied to this graph
        """
        potential_node_types = set(['MultiPath'])
        multipath_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in potential_node_types:
                multipath_nodes.append(node)
        if len(multipath_nodes) > 5:
            return False
        else:
            return True
