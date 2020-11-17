import autograd.numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pickle

import config.config as configuration
from lib.decorators import register_evolution_operators, register_crossover_operators, register_growth_operators, register_reduction_operators,register_path_reduction_operators
from lib.base_classes import EvolutionOperators as EvolutionOperators
from algorithms.speciation import Speciation
from problems.example.graph import Graph


# TODO: add hashable key to edges, to manage multidigraphs
# TODO: handle implementation assumptions
"""
Assumptions on implementation:
1. Connectors automatically scale to # of inputs / outputs
2. Connectors hold instance vars called max_in, and max_out, which describe their max number of inputs and outputs
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
        new_node_model = random.sample(self.node_models, 1)[0]()
        node_id = graph.get_next_valid_node_ID()
        # 2. Save previous edge model / other info
        interface_edge_dict = graph.edges[interface['edge']]

        # 3. Add node and edge, and connect properly
        graph.add_node(node_id, **{'model':new_node_model, 'name':new_node_model.__class__.__name__, 'lock': False})
        edge_dict = {'model':new_edge_model, 'name':new_edge_model.__class__.__name__, 'lock':False}
        # TODO: use the key attribute to pick specific edge!
        graph.remove_edge(interface['edge'][0], interface['edge'][1])

        if interface['node'] == interface['edge'][0]: # i.e. we have a NODE -> EDGE interface, and should add our components as EDGE -> NODE
            graph.add_edge(interface['node'], node_id, **edge_dict)
            graph.add_edge(node_id, interface['edge'][1], **interface_edge_dict)
        else: # we have an EDGE -> NODE interface
            graph.add_edge(interface['edge'][0], node_id, **interface_edge_dict)
            graph.add_edge(node_id, interface['node'], **edge_dict)

        # 4. TODO: call function call which updates connectors if need be

        return graph

    def possible_evo_locations(self, graph):
        """
        Returns the set of interfaces
        Nodes can be added at any interfaces (where an interface is defined as an edge and its preceding node OR the edge and its succeeding node)
        
        :param graph: graph to evolve
        
        :returns
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
    potential_node_types = set(['SinglePath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return
    
    def apply_evolution(self, graph, interface):
        pass
    
    def possible_evo_locations(self, graph):
        """
        An edge can be added between 
        """
        pass

@register_reduction_operators
@register_evolution_operators
class RemoveComponent(EvolutionOperators):
    """
    REMOVE EDGE:
    randomly select one edge (excluding the edges connected to source/sink, i.e. where laser/PD models are)
    remove the selected edge, and if there are no other paths between U, V of the selected edge then merge them
    
    Note: protected edges/nodes/connectors must not be touched. Definition of protected is given below in SwapComponent
    TODO: make rule about which model is kept

    """
    potential_node_types = set(['SinglePath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return
    
    def apply_evolution(self, graph, interface):
        pass
    
    def possible_evo_locations(self, graph):
        pass


@register_evolution_operators
class SwapComponent(EvolutionOperators):
    """
    SWAP EDGE:
    swap any model on any edge/node (except those deemed 'protected',
    i.e. source or sink models for some problems with a fixed laser,
    or in the 'sensing' problem, a phase shift element is fixed and must be kept)
    """
    potential_node_types = set(['SinglePath', 'MultiPath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return
    
    def apply_evolution(self, graph, interface):
        pass
    
    def possible_evo_locations(self, graph):
        pass
