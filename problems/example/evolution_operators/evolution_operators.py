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
        # choose a random edge to insert the node at
        edge = random.sample(graph.edges, 1)[0]
        return self.apply_evolution_at(graph, edge, verbose=verbose)

    def apply_evolution_at(self, graph, edge, save=False, verbose=False):
        # find all potential model types (all single-path nodes) to choose one from
        potential_nodes = []
        for node_type in self.potential_node_types:
            potential_nodes += list(configuration.NODE_TYPES_ALL[node_type].values())
        model = random.sample(potential_nodes, 1)[0]()

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
    
    def verify_evolution_at(self, graph, edge):
        """
        Check if the specific evolution on this edge of the graph is possible, returns bool
        """
        if edge not in set(graph.edges): # if the input edge is truly an edge
            return False

        return True # if the input is not an edge


@register_evolution_operators
@register_reduction_operators
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

        return self.apply_evolution_at(graph, node_to_remove, verbose=verbose)

    
    def apply_evolution_at(self, graph, node, save=False, verbose=False):
        # update graph connections
        if node not in graph.nodes:
            return False

        try:
            graph.add_edge(graph.pre(node)[0], graph.suc(node)[0])
        except IndexError as e:
            print(e)
            graph.draw(legend=True)
            plt.show()
        graph.remove_edge(graph.pre(node)[0], node)
        graph.remove_edge(node, graph.suc(node)[0])

        if verbose:
            print('Evolution operator: RemoveNode | Removing node {} with model {}'.format(node, graph.nodes[node]['model']))

        graph.remove_node(node)
        if graph.speciation_descriptor is not None and graph.speciation_descriptor['name'] == 'photoNEAT':
            marker = graph.speciation_descriptor['node to marker'].pop(node)
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

    def verify_evolution_at(self, graph, node):
        """
        Check if the specific evolution on this node of the graph is possible, returns bool
        """
        if node not in graph.nodes:
            return False
        return graph.nodes[node]['model'].__class__.__bases__[0].__name__ in self.potential_node_types

@register_evolution_operators
class SwapNode(EvolutionOperators):

    potential_node_types = set(['SinglePath', 'Input', 'MultiPath'])

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        """ Applies the specific evolution on the graph argument
        """

        # find all nodes which could be swapped and randomly select one
        tmp = self.collect_potential_nodes_to_swap(graph)
        potential_node_types = [node_type for (node_type, nodes) in tmp.items() if ((len(nodes) >= 1) and (len(configuration.NODE_TYPES_ALL[node_type]) > 1))]
        potential_node_type = random.sample(potential_node_types, 1)[0]

        swappable_nodes = tmp[potential_node_type]
        node_to_swap = random.sample(swappable_nodes, 1)[0]
        return self.apply_evolution_at(graph, node_to_swap, verbose=verbose)

    def apply_evolution_at(self, graph, node, save=False, verbose=False):
        pre_swap_params, *_ = copy.deepcopy(graph.extract_parameters_to_list())

        model_to_swap = graph.nodes[node]['model']
        node_type_set = set(configuration.NODE_TYPES_ALL[model_to_swap.__class__.__bases__[0].__name__].values())

        new_model = random.sample(list(node_type_set - set([model_to_swap.__class__])), 1)[0]()

        graph.nodes[node]['model'] = new_model
        graph.nodes[node]['name'] = new_model.__class__.__name__

        if verbose:
            print('Evolution operator: SwapNode | Swapping node {} from model {} to model {}'.format(node, model_to_swap, new_model))

        return graph

    def collect_potential_nodes_to_swap(self, graph):
        tmp = {potential_node_type: [] for potential_node_type in self.potential_node_types}
        for node in graph.nodes:
            if graph.nodes[node]['model'].__class__.__bases__[0].__name__ in self.potential_node_types:
                tmp[graph.nodes[node]['model'].__class__.__bases__[0].__name__].append(node)
        return tmp

    def verify_evolution(self, graph):
        """ Checks if the specific evolution on the graph is possible, returns bool

        All logic for going through graph structures and what node specifics there are, and returning a Yes/No of whether this
        evolution operator can be applied to this graph
        """

        tmp = self.collect_potential_nodes_to_swap(graph)
        flag = False
        for potential_node_type, potential_nodes in tmp.items():
            # print('Swap node check')
            # print(f"potential_node_type:{potential_node_type}, nodes:{potential_nodes}, swaps: {configuration.NODE_TYPES_ALL[potential_node_type]}")
            if (len(potential_nodes) >= 1) and (len(configuration.NODE_TYPES_ALL[potential_node_type]) > 1):
                flag = flag or True
            else:
                flag = flag or False
        return flag
    
    def verify_evolution_at(self, graph, node):
        """
        Check if the specific evolution on this edge of the graph is possible, returns bool
        """
        if node not in graph.nodes:
            return False
        # returns True if node is of a correct type, and node is not the only one of the correct type. Otherwise returns false
        node_type = graph.nodes[node]['model'].__class__.__bases__[0].__name__
        return node_type in self.potential_node_types and len(configuration.NODE_TYPES_ALL[node_type]) > 1


@register_evolution_operators
@register_growth_operators
class AddInterferometer(EvolutionOperators):

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        """ Applies the specific evolution on the graph argument
        """
        # choose two separate edges to insert the new multi-path nodes
        edge = random.sample(graph.edges, 1)[0]
        return self.apply_evolution_at(graph, edge, verbose=verbose)

    def apply_evolution_at(self, graph, edge0, save=False, verbose=False):
        # create instance of two new multi-path nodes (just 2x2 couplers for now)
        edge1 = random.sample(set(graph.edges) - set([edge0]), 1)[0]

        splitter1 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()
        splitter2 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()

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
        if graph.propagation_order.index(edge0[0]) <= graph.propagation_order.index(edge1[0]):
            edge1, edge2 = edge0, edge1
        else:
            edge1, edge2 = edge1, edge0

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
        """
        Checks if the specific evolution on the graph is possible, returns bool

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

    def verify_evolution_at(self, graph, edge):
        if edge not in set(graph.edges):
            return False
        return self.verify_evolution(graph)


@register_evolution_operators
@register_reduction_operators
@register_path_reduction_operators
class RemoveOneInterferometerPath(EvolutionOperators):
    """
    Collapses an interferometer structure to a single path (if possible)
    Note: using the complex AddInterferometer can cause this function for removing paths to fail on occasion
    """
    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        return self.apply_evolution_at(graph, None, verbose=verbose) # node is passed as None bc it's the easiest way to avoid legacy breaking things rn

    def apply_evolution_at(self, graph, node, verbose=False, save=False):
        """ Applies the specific evolution on the graph argument
        """
        if save:
            with open (f'remove_interferometer_pre_graph.pkl', 'wb') as handle:
                pickle.dump(graph, handle)
        # convert graph to a simple (undirected, non-hyper, simple graph) to find cycles
        cycles = self.get_graph_cycles(graph)

        # we choose a random cycle that we will consider. This cycle must contain 'node' (precondition is that node is part of a cycle)
        if node is not None: # select an arbitrary cycle which 'node' is part of. The if statement is for backwards compatible of a random evolution application
            cycles = [cycle for cycle in cycles if node in cycle]
        cycle = random.sample(cycles, 1)[0] # chooses random cycle to remove part of
        # tmp_ = [len(cycle) for cycle in cycles]
        # cycle = cycles[tmp_.index(max(tmp_))]

        if save:
            with open (f'cycle_to_remove_interferometer_graph.pkl', 'wb') as handle:
                pickle.dump(cycle, handle)
            print(f'node to remove path around: {node}')
            print(f'cycle to remove: {cycle}')

        # use the old propagation order to ensure physicality when placing new nodes
        propagation_order = graph.propagation_order

        # find the source/sink node of the cycle by comparing to the propagation order (first in order is the source, last is sink)
        propagation_index = [propagation_order.index(node) for node in cycle]
        source_node = cycle[np.argmin(propagation_index)]
        sink_node = cycle[np.argmax(propagation_index)]
        if save:
            print(f'propagation index: {propagation_index}')
            print(f'source node: {source_node}')
            print(f'sink node: {sink_node}')

        # choose one branch to remain in the graph, we will delete the others
        branch_to_keep = random.sample(graph.suc(source_node), 1)[0]

        if branch_to_keep == sink_node:
            branch_to_keep = None
        
        if save:
            with open (f'branch_to_keep_remove_interferometer_graph.pkl', 'wb') as handle:
                pickle.dump(cycle, handle)
            print(f'branch to keep: {branch_to_keep}')
    
        # check what nodes/edges to keep and remove
        paths_from_source_to_sink = list(nx.all_simple_paths(graph, source_node, sink_node))
        paths_to_remove = [path for path in paths_from_source_to_sink if branch_to_keep not in path]
        nodes_to_remove = set([item for sublist in paths_to_remove for item in sublist])
        paths_to_keep = [path for path in paths_from_source_to_sink if branch_to_keep in path]
        nodes_to_keep = set([item for sublist in paths_to_keep for item in sublist]) - set([source_node, sink_node])
        if save:
            print(f'paths source to sink: {paths_from_source_to_sink}')
            print(f'paths to remove: {paths_to_remove}')
            print(f'nodes to remove: {nodes_to_remove}')
            print(f'paths to keep: {paths_to_keep}')
            print(f'nodes to keep: {nodes_to_keep}')
            graph.draw()
            plt.show()

        # connect graph back together deal with the possibilities of removing a whole loop
        if not nodes_to_keep:
            if save:
                print(f'not keeping nodes, adding edge: ({graph.pre(source_node)[0]}, {graph.suc(sink_node)[0]})')
            graph.add_edge(graph.pre(source_node)[0], graph.suc(sink_node)[0])
        else:
            tmp = list(nodes_to_keep)
            if save:
                print(f'nodes to keep: {tmp}')
            propagation_index = [propagation_order.index(node) for node in tmp]
            new_source_node = tmp[np.argmin(propagation_index)]
            new_sink_node = tmp[np.argmax(propagation_index)]

            graph.add_edge(graph.pre(source_node)[0], new_source_node)
            graph.add_edge(new_sink_node, graph.suc(sink_node)[0])
            if save:
                print(f'adding edges: ({graph.pre(source_node)[0]},{new_source_node}) and ({new_sink_node}, {graph.suc(sink_node)[0]})')


        # remove the nodes that should be deleted
        graph.remove_nodes_from(nodes_to_remove)

        # HACKY fix for rare issue of floating/unattached nodes, which can occur with nested interferometers
        flag = True
        while flag:

            try:
                graph.assert_number_of_edges()
                break # no issues, can continue on
            except TypeError as E:
                floating_nodes = []
                for node in graph.nodes:
                    try:
                        graph.nodes[node]['model'].assert_number_of_edges(graph.get_in_degree(node), graph.get_out_degree(node))
                    except TypeError as E:
                        floating_nodes.append(node)
                if len(floating_nodes) == 2: # if there's two unconnected ndoes, just try stitching them together
                    edge_try1 = [(floating_nodes[0], floating_nodes[1])]
                    edge_try2 = [(floating_nodes[1], floating_nodes[0])]
                    try:
                        graph.add_edges_from(edge_try1)
                        graph.assert_number_of_edges()
                        break
                    except:
                        graph.remove_edges_from(edge_try1)
                        pass
                    try:
                        graph.add_edges_from(edge_try2)
                        graph.assert_number_of_edges()
                        break
                    except:
                        graph.remove_edges_from(edge_try2)
                        pass

                if save:
                    print(f'------------Floating nodes:{floating_nodes}')
                if floating_nodes is not None:
                    graph.remove_nodes_from(floating_nodes)
            # end of gross HACKY fix

        if graph.speciation_descriptor is not None and graph.speciation_descriptor['name'] == 'photoNEAT':
            for node in nodes_to_remove:
                marker = graph.speciation_descriptor['node to marker'].pop(node)
                graph.speciation_descriptor['marker to node'].pop(marker)            

        if verbose:
            str_info = 'Source node {} | Sink node {} | Branch to keep {} | Nodes to delete {}'.format(source_node, sink_node, branch_to_keep, nodes_to_remove)
            print('Evolution operator: RemoveOneInterferometerPath | {}'.format(str_info))
        
        if save:
            graph.draw()
            plt.show()

        return graph

    def verify_evolution(self, graph):
        """ Checks if the specific evolution on the graph is possible, returns bool

        All logic for going through graph structures and what node specifics there are, and returning a Yes/No of whether this
        evolution operator can be applied to this graph
        """
        cycles = self.get_graph_cycles(graph)

        if len(cycles) > 0:
            return True
        else:
            return False
    
    def verify_evolution_at(self, graph, node):
        if node not in graph.nodes:
            return False

        cycles = self.get_graph_cycles(graph)
        for cycle in cycles:
            if node in cycle:
                return True
        return False
    
    def get_graph_cycles(self, graph):
        # check to see if there is a cycle which can be collapses
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
        return cycles


# @register_evolution_operators
# @register_growth_operators
class AddInterferometerSimple(EvolutionOperators):

    def __init__(self, **attr):
        super().__init__(**attr)
        return

    def apply_evolution(self, graph, verbose=False):
        """ Applies the specific evolution on the graph argument
        """
        # edge to add interferometer in
        edge = random.sample(graph.edges, 1)[0]
        return self.apply_evolution_at(graph, edge, verbose=verbose)
    
    def apply_evolution_at(self, graph, edge, verbose=False, save=False):
        # two new multipath splitters (just 2x2 couplers for now)
        splitter1 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()
        splitter2 = configuration.NODE_TYPES_ALL['MultiPath']['VariablePowerSplitter']()

        # insert a random single-path node
        potential_node_types = set(['SinglePath'])
        potential_nodes = []
        for node_type in potential_node_types:
            potential_nodes += list(configuration.NODE_TYPES_ALL[node_type].values())
        new_nonsplitter = random.sample(potential_nodes, 1)[0]()

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
    
    def verify_evolution_at(self, graph, edge):
        if edge not in set(graph.edges):
            return False

        return self.verify_evolution(graph)


@register_crossover_operators
class SinglePointCrossover(EvolutionOperators):
    """
    A simple crossover operator which allows crossovers ONLY at bridges in the graph
    Modifies graph0 and graph1 (in that it DOES NOT do a deepcopy of their nodes / edges and makes it undirected)
    """
    def __init__(self, **attr):
        super().__init__(**attr)

    def apply_evolution(self, graph0, graph1, verbose=False, debug=False):
        # TODO: pick bridge using NEAT historical marker methods
        if debug:
            print(f'graph0\nnodes: {graph0.nodes}\nedges: {graph0.edges}')
            print(f'graph1\nnodes: {graph1.nodes}\nedges: {graph1.edges}\n\n')

        # 1. Pick a bridge on each graph. This is the edge across which we'll split the graphs
        bridge0 = random.sample(list(nx.bridges(nx.Graph(graph0))), 1)[0]
        bridge1 = random.sample(list(nx.bridges(nx.Graph(graph1))), 1)[0]

        bridge0_model = None
        try:
            bridge0_model = graph0.edges[bridge0[0], bridge0[1]]['model']
        except:
            pass

        if debug:
            print(f'bridge0: {bridge0}, model0: {bridge0_model}')
            print(f'bridge1: {bridge1}')

        # 2. Remove the edges bridge0 and bridge1 from the graphs. Like this, we get the 4 components we want to crossover
        # try and except is bc we get the bridge from an undirected graph, so the order could be flipped
        try:
            graph0.remove_edge(bridge0[0], bridge0[1])
        except nx.exception.NetworkXError:
            graph0.remove_edge(bridge0[1], bridge0[0])
        
        try:
            graph1.remove_edge(bridge1[0], bridge1[1])
        except nx.exception.NetworkXError:
            graph1.remove_edge(bridge1[1], bridge1[0])

        # 3. Grab the relevant components
        graph0_comps = [c for c in nx.weakly_connected_components(graph0)]
        graph1_comps = [c for c in nx.weakly_connected_components(graph1)]

        graph0_A = graph0_comps[0]
        graph0_B = graph0_comps[1]
        graph1_A = graph1_comps[0]
        graph1_B = graph1_comps[1]

        if debug:
            print(f'graph0_A: {graph0_A}')
            print(f'graph0_B: {graph0_B}')
            print(f'graph1_A: {graph1_A}')
            print(f'graph1_B: {graph1_B}')

        # 4. Create the children
        # 4.1 Create the nodes
        graph0_start_nodes, graph0_end_nodes = (graph0_A, graph0_B) if 0 in graph0_A else (graph0_B, graph0_A)
        graph1_start_nodes, graph1_end_nodes = (graph1_A, graph1_B) if 0 in graph1_A else (graph1_B, graph1_A)

        def new_node_num(node_num, offset):
            if node_num == 0 or node_num == -1:
                return node_num
            return node_num + offset
            
        child0_nodes = {node_num: graph0.nodes[node_num]['model'] for node_num in graph0_start_nodes}
        try:
            node_num_offset0 = max(graph0_start_nodes) + 1 - min(graph1_end_nodes - set([-1]))
        except ValueError: # if -1 is the only value in graph1_end_nodes, offset will not matter
            node_num_offset0 = 0

        if debug:
            print(f'node_num_offset0: {node_num_offset0}')
        child0_nodes.update({new_node_num(node_num, node_num_offset0): graph1.nodes[node_num]['model'] for node_num in graph1_end_nodes})

        child1_nodes = {node_num: graph1.nodes[node_num]['model'] for node_num in graph1_start_nodes}
        try:
            node_num_offset1 = max(graph1_start_nodes) + 1 - min(graph0_end_nodes - set([-1]))
        except ValueError: # if -1 is the only value in graph0_end_nodes, offset will not matter
            node_num_offset1 = 0
        if debug:
            print(f'node_num_offset1: {node_num_offset1}')
        child1_nodes.update({new_node_num(node_num, node_num_offset1): graph0.nodes[node_num]['model'] for node_num in graph0_end_nodes})

        if debug:
            print(f'\nchild0 nodes: {child0_nodes.keys()}')
            print(f'\nchild1 nodes: {child1_nodes.keys()}')

        # 4.2 Create the edges
        child0_edges = []
        child1_edges = []

        def new_edge(edge, offset, graph):
            node_in = edge[0] if (edge[0] == 0 or edge[0] == -1) else edge[0] + offset
            node_out = edge[1] if (edge[1] == 0 or edge[1] == -1) else edge[1] + offset
            try:
                return (node_in, node_out, graph.edges[edge]['model'])
            except KeyError:
                return (node_in, node_out)
    
        for edge in graph0.edges:
            if edge[0] in graph0_start_nodes:
                child0_edges.append(new_edge(edge, 0, graph0))
            else:
                child1_edges.append(new_edge(edge, node_num_offset1, graph0))
        
        for edge in graph1.edges:
            if edge[0] in graph1_start_nodes:
                child1_edges.append(new_edge(edge, 0, graph1))
            else:
                child0_edges.append(new_edge(edge, node_num_offset0, graph1))

        # 4.3 Add the new bridge edges
        (child0_connect_start, child1_connect_end) = (bridge0[0], bridge0[1]) if bridge0[0] in graph0_start_nodes else (bridge0[1], bridge0[0])
        (child1_connect_start, child0_connect_end) = (bridge1[0], bridge1[1]) if bridge1[0] in graph1_start_nodes else (bridge1[1], bridge1[0])

        if debug:
            print(f'child0 goes from (pre node name update): {(child0_connect_start, child0_connect_end)}')
            print(f'child1 goes from (pre node name update): {(child1_connect_start, child1_connect_end)}')

        child0_bridge = (child0_connect_start, new_node_num(child0_connect_end, node_num_offset0))
        child1_bridge = (child1_connect_start, new_node_num(child1_connect_end, node_num_offset1))

        if bridge0_model is not None:
            child0_bridge = (child0_bridge[0], child0_bridge[1], bridge0_model)
            child1_bridge = (child1_bridge[0], child1_bridge[1], bridge0_model)
        # child0_bridge = (child0_connect_start, child0_connect_end) if bridge0_model is None else (child0_connect_start, child0_connect_end, bridge0_model)
        # child1_bridge = (child1_connect_start, child1_connect_end) if bridge0_model is None else (child1_connect_start, child1_connect_end, bridge0_model)

        # child0_edges.append(new_edge_num(child0_bridge, node_num_offset0))
        # child1_edges.append(new_edge_num(child1_bridge, node_num_offset1))
        child0_edges.append(child0_bridge)
        child1_edges.append(child1_bridge)

        if debug:
            print(f'child0 edges: {child0_edges}')
            print(f'child1 edges: {child1_edges}')

        # 5. Make children
        child0 = Graph(nodes=child0_nodes, edges=child0_edges, propagate_on_edges=graph0.propagate_on_edges, coupling_efficiency=graph0.coupling_efficiency)
        child1 = Graph(nodes=child1_nodes, edges=child1_edges, propagate_on_edges=graph0.propagate_on_edges, coupling_efficiency=graph0.coupling_efficiency)
        if debug:
            print(f'child0_nodes, full: {child0_nodes}')
            for node in child0.nodes:
                print(f'child0: {child0.nodes[node]}')
            # child0.draw()
            # child1.draw()
            # plt.show()
        
        child0.assert_number_of_edges()
        child1.assert_number_of_edges()

        if verbose:
            print(f'Crossover operator: SinglePointCrossover | Graphs split and crossover at bridge0: {bridge0}, bridge1: {bridge1}')

        return child0, child1

    def verify_evolution(self, graph0, graph1):
        """
        Should not make modifications to graph0 and graph1
        """
        # return True # I think it HAS to always work actually, since our sources / detectors are single connection
        # undirected0 = graph0.to_undirected()
        # undirected1 = graph1.to_undirected()
        undirected0 = nx.Graph(graph0)
        undirected1 = nx.Graph(graph1)
        return nx.has_bridges(undirected0) and nx.has_bridges(undirected1)


