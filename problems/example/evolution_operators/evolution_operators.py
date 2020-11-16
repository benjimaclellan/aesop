import autograd.numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# TODO: remove once debugging complete
import copy 
from autograd.numpy.numpy_boxes import ArrayBox
# -------------------------------

import config.config as configuration
from lib.decorators import register_evolution_operators, register_crossover_operators, register_growth_operators, register_reduction_operators,register_path_reduction_operators
from lib.base_classes import EvolutionOperators as EvolutionOperators
from algorithms.speciation import Speciation
from problems.example.graph import Graph

@register_growth_operators
@register_evolution_operators
class AddSeriesComponent(EvolutionOperators):
    """
    ADD EDGE + NODE (series concatenation)
    randomly select one 'interface' (i.e. start or end of one edge), excluding edges the start/end at the source or sink nodes
    add a random component model to the edge, the node will be a 1x1 connector

    'interface' = (node, edge) where node is either the pre or post node of edge
    """
    potential_node_types = set(['SinglePath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return
    
    def apply_evolution(self, graph, interface):
        pass
    
    def possible_evo_locations(self, graph):
        """
        Any interface other than 
        """
        pass


@register_growth_operators
@register_evolution_operators
class AddParallelComponent(EvolutionOperators):
    """
    ADD EDGE (parallel):
    select two nodes (randomly, without replacement, excluding source/sink nodes) and add an edge between them (in the direction that maintains status as a DAG)
    the model on the nodes may have to change, which is chosen randomly (i.e. randomly choose from NxN beamsplitter or frequency-dependent splitter, depending on what models are in the library)
    
    Note: we currently only consider beamsplitters
    """
    potential_node_types = set(['SinglePath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return
    
    def apply_evolution(self, graph, interface):
        pass
    
    def possible_evo_locations(self, graph):
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
