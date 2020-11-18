import autograd.numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import itertools

import config.config as configuration
from lib.decorators import register_evolution_operators, register_crossover_operators, register_growth_operators, register_reduction_operators,register_path_reduction_operators
from lib.base_classes import EvolutionOperators as EvolutionOperators
from algorithms.speciation import Speciation
from problems.example.graph import Graph


# TODO: add hashable key to edges, to manage multidigraphs
# TODO: add verbose mode
# TODO: handle implementation assumptions
"""
Assumptions on implementation:
1. Connectors automatically scale to # of inputs / outputs (with a potential function call)
2. Connectors models hold instance vars _range_input_edges, which describe their max number of inputs and outputs (this is true for now)
3. Models are protected: nodes/edges themselves are not. I.e. nodes / edges have a property s.t. graph.nodes[node]['model'].protected is a boolean (true for now)
"""


@register_growth_operators
@register_evolution_operators
class AddSeriesComponent(EvolutionOperators):
    """
    ADD EDGE + NODE (series concatenation)
    randomly select one 'interface' (i.e. start or end of one edge), excluding edges the start/end at the source or sink nodes
    add a random component model to the edge, the node will be a 1x1 connector

    'interface' = (node, edge) where node is either the pre or post node of edge
    """

    def __init__(self, **attr):
        super().__init__(**attr)
        return
    
    def apply_evolution(self, graph, interface):  
        # 1. Pick/create new models
        new_edge_model = random.sample(self.edge_models, 1)[0]()
        new_edge_model_name = new_edge_model.__class__.__name__
        new_node_model = random.sample(self.node_models, 1)[0]()
        new_node_model_name = new_node_model.__class__.__name__
        node_id = graph.get_next_valid_node_ID()
        # 2. Save previous edge model / other info
        interface_edge_dict = graph.edges[interface.edge]

        # 3. Add node and edge, and connect properly
        graph.add_node(node_id, **{'model':new_node_model, 'name': new_node_model_name, 'lock': False})
        edge_dict = {'model':new_edge_model, 'name':new_edge_model_name, 'lock':False}
        # TODO: use the key attribute to pick specific edge!
        graph.remove_edge(interface.edge[0], interface.edge[1])

        if interface.node == interface.edge[0]: # i.e. we have a NODE -> EDGE interface, and should add our components as EDGE -> NODE
            graph.add_edge(interface.node, node_id, **edge_dict)
            graph.add_edge(node_id, interface.edge[1], **interface_edge_dict)
        else: # we have an EDGE -> NODE interface
            graph.add_edge(interface.edge[0], node_id, **interface_edge_dict)
            graph.add_edge(node_id, interface.node, **edge_dict)

        # 4. TODO: call function call which updates connectors if need be

        if self.verbose:
            print(f"Evo Op: AddSeriesComponent | Added edge ({new_edge_model_name}) and node ({new_node_model_name}) \
                    at the interface between node {interface.node} and edge {interface.edge}")

        return graph

    def possible_evo_locations(self, graph):
        """
        Returns the set of interfaces
        Nodes can be added at any interfaces (where an interface is defined as an edge and its preceding node OR the edge and its succeeding node)
        
        :param graph: graph to evolve
        
        :returns the interfaces (which are possible evo locations)
        """
        return graph.interfaces


@register_growth_operators
@register_evolution_operators
class AddParallelComponent(EvolutionOperators):
    """
    ADD EDGE (parallel):
    select two nodes (randomly, without replacement, excluding source/sink nodes) and add an edge between them (in the direction that maintains status as a DAG)
    the model on the nodes may have to change, which is chosen randomly (i.e. randomly choose from NxN beamsplitter or frequency-dependent splitter, depending on what models are in the library)
    
    Note: we currently only consider beamsplitters
    
    TODO: consider input and output caps
    """
    def __init__(self, **attr):
        super().__init__(**attr)
        return
    
    def apply_evolution(self, graph, node_tuple):
        new_edge_model = random.sample(self.edge_models, 1)[0]()
        new_edge_model_name = new_edge_model.__class__.__name__
        graph.add_edge(node_tuple[0], node_tuple[1], **{'model':new_edge_model, 'name':new_edge_model_name, 'lock':False})
        
        if self.verbose:
            print(f"Evo Op: AddParallelComponent | Added edge ({new_edge_model_name}) between nodes {node_tuple[0]} and {node_tuple[1]}")
        return graph
    
    def possible_evo_locations(self, graph):
        """
        An edge can be added between any two nodes ASSUMING
        1. The nodes aren't terminal
        2. The nodes aren't protected
        3. The nodes haven't maxed out their in-out degree
        """
        # TODO: verify that the order is provided correctly (i.e. all tuples are in the correct topological sort). I think it is...
        nodes = [node for node in nx.algorithms.dag.topological_sort(graph) \
                                                if (graph.in_degree[node] != 0 and \
                                                    graph.out_degree[node] != 0 and \
                                                    not graph.nodes[node]['model'].protected and \
                                                    graph.in_degree[node] < graph.nodes[node]['model']._range_input_edges[1] and \
                                                    graph.out_degree[node] < graph.nodes[node]['model']._range_input_edges[1])]
        return list(itertools.combinations(nodes, 2))

@register_reduction_operators
@register_evolution_operators
class RemoveComponent(EvolutionOperators):
    """
    REMOVE EDGE:
    randomly select one interface (excluding the interfaces of source/sink, i.e. where laser/PD models are)
    remove the selected interface's edge, and if there are no other paths between U, V of the selected edge then merge them
    while keeping the connector-type of the interface's node.

    Note: only the edge of the interface matters, UNLESS there are no other paths between U, V
    Note: protected edges/nodes must not be touched. Definition of protected is given below in SwapComponent
    TODO: make rule about which model is kept

    """
    def __init__(self, **attr):
        super().__init__(**attr)
        return
    
    def apply_evolution(self, graph, interface):
        # 1. Remove the edge
        # TODO: fix such that the edges have a hashable key
        edge_model_name = graph.edges[interface.edge]['model'].__class__.__name__
        graph.remove_edge(interface.edge[0], interface.edge[1])

        merged_nodes = False
        # 2. check whether we need to merge nodes
        if not nx.algorithms.shortest_paths.generic.has_path(graph, interface.edge[0],  interface.edge[1]):
            # 3. If not, merge U, V keeping the datatype of U if interface.node == U, keeping the datatype of V otherwise
            # note: the merge leaves the label of U in the graph, regardless of which model we keep
            save_node_dict = graph.nodes[interface.node]
            nx.algorithms.minors.contracted_nodes(graph,  interface.edge[0],  interface.edge[1], self_loops=False, copy=False)
            graph.nodes[interface.edge[0]].update(**save_node_dict)
            # TODO: function call to update the connector model on U
            merged_nodes = True
        
        if self.verbose:
            message = f"Evo Op: RemoveComponent | Removed edge {interface.edge} ({edge_model_name})"
            if merged_nodes:
                message += f" | Merged nodes { interface.edge[0]} and {interface.edge[1]} into a single {save_node_dict['model'].__class__.__name__}"
            print(message)

        return graph
    
    def possible_evo_locations(self, graph):
        interfaces = [interface for interface in graph.interfaces if \
                      (graph.in_degree[interface.edge[0]] != 0 and \
                       graph.out_degree[interface.edge[1]] != 0 and \
                       not graph.edges[interface.edge]['model'].protected)]
        return interfaces


@register_evolution_operators
class SwapComponent(EvolutionOperators):
    """
    SWAP EDGE:
    swap any model on any edge/node (except those deemed 'protected',
    i.e. source or sink models for some problems with a fixed laser,
    or in the 'sensing' problem, a phase shift element is fixed and must be kept)
    """
    def __init__(self, **attr):
        super().__init__(**attr)
        return
    
    def apply_evolution(self, graph, node_edge):
        if node_edge in graph.nodes: # swap node
            initial_model_name = graph.nodes[node_edge]['model'].__class__.__name__
            node_set = self.node_models - set([graph.nodes[node_edge]['model'].__class__])
            new_model_name = self._swap_if_possible(graph, node_edge, node_set, True)
        else: # swap edge
            initial_model = graph.edges[node_edge]['model'].__class__
            initial_model_name = initial_model.__name__
            if graph.nodes[node_edge[0]]['model']._node_type == 'source node':
                source_set = self.source_models - set([initial_model])
                new_model_name = self._swap_if_possible(graph, node_edge, source_set, False)
            elif graph.nodes[node_edge[1]]['model']._node_type == 'sink node':
                sink_set = self.sink_models - set([initial_model])
                new_model_name = self._swap_if_possible(graph, node_edge, sink_set, False)
            else:
                edge_set = self.edge_models - set([initial_model])
                new_model_name = self._swap_if_possible(graph, node_edge, source_set, False)

        if self.verbose:
            print(f"Evo Op: SwapComponent | Swapped model at {node_edge} from {initial_model_name} to {new_model_name}")

        return graph

    def _swap_if_possible(self, graph, node_edge, swap_set, is_node):
        """
        Swaps the model at location node_edge in graph, with a value from the swap_set. Returns the name of the new model class
        (or name of the old model class, if there is no new class. This happens if swap_set contains only the model already at node_edge)
        """
        if len(swap_set) != 0:
            if is_node:
                graph.nodes[node_edge]['model'] = random.sample(swap_set, 1)[0]()
                return graph.nodes[node_edge]['model'].__class__.__name__
            else:
                graph.edges[node_edge]['model'] = random.sample(swap_set, 1)[0]()
                return graph.edges[node_edge]['model'].__class__.__name__
        elif self.verbose:
            print(f'WARNING: no valid source models to swap with, swap was not executed')
            if is_node:
                return graph.nodes[node_edge]['model'].__class__.__name__
            else:
                return graph.edges[node_edge]['model'].__class__.__name__

    def possible_evo_locations(self, graph):
        edges = [edge for edge in graph.edges if not graph.edges[edge]['model'].protected]
        nodes = [node for node in graph.nodes if (not graph.nodes[node]['model'].protected and \
                                                      graph.in_degree[node] != 0 and \
                                                      graph.out_degree[node] != 0)]
        return nodes + edges
