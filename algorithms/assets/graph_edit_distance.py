import networkx as nx
import config.config as config 

def graph_kernel_map_to_nodetypes(_graph):
    """
    NOT SUPPORTED AFTER GRAPH ENCODING CHANGE.
    A pre-processing step to collapse nodes to their model types.
    :param _graph:
    :return:
    """
    graph_relabelled = nx.relabel_nodes(_graph, {node: _graph.nodes[node]['model'].node_acronym for node in _graph.nodes})
    all_node_relabels = []
    for node_subtypes in config.NODE_TYPES_ALL.values():
        for node_subtype in node_subtypes.values():
            all_node_relabels.append(node_subtype.node_acronym)
    graph_relabelled.add_nodes_from(all_node_relabels)
    return graph_relabelled

def similarity_full_ged(g1, g2):
    """
    Measures the Graph Edit Distance similarity between two graphs exactly. Can be slow, it is suggested use the
    approximate (reduced) method instead
    :param _graph1: Graph object
    :param _graph2: Graph object
    :return: similarity (integer number of steps needed to transform Graph 1 to Graph 2
    """
    sim = nx.algorithms.similarity.graph_edit_distance(g1, g2,
                                                       edge_subst_cost=edge_match,
                                                       node_subst_cost=node_match,
                                                       upper_bound=30.0,
                                                       timeout=10.0,
                                                       )
    return sim

def similarity_reduced_ged(g1, g2):
    """
    Approximated the Graph Edit Distance similarity between two graphs exactly.
    :param _graph1: Graph object
    :param _graph2: Graph object
    :return: similarity (integer number of steps needed to transform Graph 1 to Graph 2
    """
    ged_approx = nx.algorithms.similarity.optimize_graph_edit_distance(g1, g2,
                                                                       edge_subst_cost=edge_match,
                                                                       node_subst_cost=node_match,
                                                                       upper_bound=30.0,
                                                                       )
    sim =  next(ged_approx) # faster, but less accurate
    return sim

def edge_match(e1, e2):
    # provides the comparison for the cost of substituting two edges or two nodes in the GED calculation
    if type(e1['model']) == type(e2['model']):
        cost = 0.0
    else:
        cost = 2.0
    return cost

def node_match(e1, e2):
    # provides the comparison for the cost of substituting two edges or two nodes in the GED calculation
    if type(e1['model']) == type(e2['model']):
        cost = 0.0
    else:
        cost = 0.5
    return cost