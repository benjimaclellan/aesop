import networkx as nx
import random
import string


class AccessGraph(object):

    def __init__(self):
        return

    # @classmethod
    # def from_asope_graph(cls, asope_graph):
    #     G = cls()
    #     # asope_graph = nx.convert_node_labels_to_integers(asope_graph, first_label=0, ordering='default', label_attribute=None)
    #     for i, node in enumerate(asope_graph.nodes):
    #         G.add_node(node, test_attr=asope_graph.nodes[node]['model'].node_acronym)
    #     for j, edge in enumerate(asope_graph.edges):
    #         G.add_edge(edge[0], edge[1], test_attr='edge_test_attr')
    #     return G

    @staticmethod
    def get_graph_positions(graph):
        return nx.kamada_kawai_layout(graph)

    @staticmethod
    def init_dict():
        edge_dict = dict(start=[],
                         end=[],
                         )
        node_dict = dict(index=[],
                         type=[],
                         test_attr=[]
                         )
        return node_dict, edge_dict

    @staticmethod
    def export_as_dict(graph):
        edge_dict = dict(start=[edge[0] for edge in graph.edges],
                         end=[edge[1] for edge in graph.edges],
                         component=[graph.edges[edge]['model'].node_acronym for edge in graph.edges],
                         edge=[edge for edge in graph.edges]
                         )

        node_dict = dict(index=[node for node in graph.nodes],
                         type=[random.choice(string.ascii_lowercase) for _ in graph.nodes],
                         component=[graph.nodes[node]['model'].node_acronym for node in graph.nodes]
                         )

        return node_dict, edge_dict

    @staticmethod
    def get_layout(graph):
        graph_layout = nx.kamada_kawai_layout(graph)
        return graph_layout
