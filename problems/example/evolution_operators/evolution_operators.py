import numpy as np
import random
import networkx as nx


import config.config as configuration
from lib.decorators import register_evolution_operators
from lib.base_classes import EvolutionOperators as EvolutionOperators


@register_evolution_operators
class AddNode(EvolutionOperators):

    potential_node_types = set(['SinglePath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph):
        """ Applies the specific evolution on the graph argument
        """


        potential_nodes = []
        for node_type in self.potential_node_types:
            potential_nodes += list(configuration.NODE_TYPES_ALL[node_type].values())

        model = random.sample(potential_nodes, 1)[0]()

        edge = random.sample(graph.edges, 1)[0]

        node_label = min(set(range(min(graph.nodes), max(graph.nodes)+2)) - set(graph.nodes))
        graph.add_node(node_label, **{'model': model, 'name': 'FEATURE NOT IMPLEMENTED', 'lock': False})

        graph.add_edge(edge[0], node_label)
        graph.add_edge(node_label, edge[1])
        graph.remove_edge(edge[0], edge[1])

        # print('inside add_node() now. adding node {} with model {}'.format(node_label, model))
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

    def apply_evolution(self, graph):
        """ Applies the specific evolution on the graph argument
        """

        potential_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in self.potential_node_types:
                potential_nodes.append(node)
        node_to_remove = random.sample(potential_nodes, 1)[0]

        # print('inside remove_node() now. removing node {} of model {}'.format(node_to_remove,
        #                                                                       graph.nodes[node_to_remove]['model']))

        graph.add_edge(graph.pre(node_to_remove)[0], graph.suc(node_to_remove)[0])
        graph.remove_edge(graph.pre(node_to_remove)[0], node_to_remove)
        graph.remove_edge(node_to_remove, graph.suc(node_to_remove)[0])

        graph.remove_node(node_to_remove)

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

    def apply_evolution(self, graph):
        """ Applies the specific evolution on the graph argument
        """

        swappable_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in self.potential_node_types:
                swappable_nodes.append(node)
        node_to_swap = random.sample(swappable_nodes, 1)[0]

        potential_new_nodes = []
        for node_type in self.potential_node_types:
            potential_new_nodes += list(configuration.NODE_TYPES_ALL[node_type].values())

        new_model = random.sample(potential_new_nodes, 1)[0]()
        graph.nodes[node_to_swap]['model'] = new_model

        # print('inside swap_node() now. swapping node {} with model {}'.format(node_to_swap, new_model))

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


# @register_evolution_operators
class AddInterferometer(EvolutionOperators):

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph):
        """ Applies the specific evolution on the graph argument
        """

        splitter1 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()
        splitter2 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()

        edge = random.sample(graph.edges, 1)[0]

        node_labels = set(range(min(graph.nodes), max(graph.nodes) + 3)) - set(graph.nodes)
        label1 = min(node_labels)
        label2 = min(node_labels - set([label1]))

        graph.add_node(label1, **{'model': splitter1, 'name': 'FEATURE NOT IMPLEMENTED', 'lock': False})
        graph.add_node(label2, **{'model': splitter2, 'name': 'FEATURE NOT IMPLEMENTED', 'lock': False})

        graph.remove_edge(edge[0], edge[1])

        graph.add_edge(edge[0], label1)
        graph.add_edge(label2, edge[1])

        graph.add_edge(label1, label2)
        graph.add_edge(label1, label2)

        # print('inside add_interferometer() on edge'.format(edge))

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



# @register_evolution_operators
class RemoveOneInterferometerPath(EvolutionOperators):

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph):
        """ Applies the specific evolution on the graph argument
        """

        # having a cycle implies a multipath
        cycles = nx.find_cycle(nx.to_undirected(graph))
        nodes_in_cycle = set([edge[0] for edge in cycles] + [edge[1] for edge in cycles])

        # find the sink and source nodes of the
        source_node, sink_node = None, None
        for node in nodes_in_cycle:
            for neighbour in graph.pre(node):
                if neighbour not in nodes_in_cycle:
                    source_node = node
            for neighbour in graph.suc(node):
                if neighbour not in nodes_in_cycle:
                    sink_node = node

        paths_from_source_to_sink = list(nx.all_simple_paths(graph, source_node, sink_node))

        assert len(paths_from_source_to_sink) >= 2

        path_ind = random.randint(0, len(paths_from_source_to_sink) - 1)
        path_to_remove = paths_from_source_to_sink[path_ind]
        path_to_keep = paths_from_source_to_sink[(path_ind + 1) % len(paths_from_source_to_sink)][1:-1]

        if not path_to_keep:
            for pre in graph.pre(source_node):
                for suc in graph.suc(sink_node):
                    graph.add_edge(pre, suc)

        else:
            # connect remaining nodes to rest of graph, assume splitter has one input
            for pre in graph.pre(source_node):
                graph.add_edge(pre, path_to_keep[0])
                # print('connecting {} and  {}'.format(pre, path_to_keep[0]))

            for suc in graph.suc(sink_node):
                graph.add_edge(path_to_keep[-1], suc)
                # print('connecting {} and  {}'.format(path_to_keep[-1], suc))

        # remove edges in the path
        for ind in range(len(path_to_remove) - 1):
            graph.remove_edge(path_to_remove[ind], path_to_remove[ind + 1])
        # remove the nodes in the path
        for node in path_to_remove:
            graph.remove_node(node)

        print('inside remove-half-an-interferometer() on cycle'.format(nodes_in_cycle))
        return graph

    def verify_evolution(self, graph):
        """ Checks if the specific evolution on the graph is possible, returns bool

        All logic for going through graph structures and what node specifics there are, and returning a Yes/No of whether this
        evolution operator can be applied to this graph
        """
        # potential_node_types = set(['MultiPath'])
        # multipath_nodes = []
        # for node in graph.nodes:
        #     if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in potential_node_types:
        #         multipath_nodes.append(node)
        # if len(multipath_nodes) > 2:
        #     return True
        # else:
        #     return False
        try:
            cycles = nx.find_cycle(nx.to_undirected(graph))
            print('TRUE')
            return True
        except:
            print('FALSE')
            return False