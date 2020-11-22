import networkx as nx
import random
import string

class VisualGraph(nx.Graph):

    @classmethod
    def from_asope_graph(cls, asope_graph):
        G = cls()
        # asope_graph = nx.convert_node_labels_to_integers(asope_graph, first_label=0, ordering='default', label_attribute=None)
        for i, node in enumerate(asope_graph.nodes):
            G.add_node(node, test_attr=asope_graph.nodes[node]['model'].node_acronym)
        for j, edge in enumerate(asope_graph.edges):
            G.add_edge(edge[0], edge[1], test_attr='edge_test_attr')
        return G

    @classmethod
    def create_empty_graph(cls):
        G = cls()
        n = 2
        for node in range(0, n):
            G.add_node(node, test_attr='test')
        for u, v in zip(range(0,n-1), range(1,n)):
            G.add_edge(u, v, test_attr='test')
        return G

    def export_as_dict(self):
        edge_dict = dict(start=[edge[0] for edge in self.edges],
                         end=[edge[1] for edge in self.edges],
                         )
        # for attr in ['test_attr']:
        #     edge_dict[attr] = ['q' for _ in self.edges]
        #                  test_attr=[self.edges[edge]['test_attr'] for edge in self.edges],
        #                  )

        node_indices = [node for node in self.nodes]
        node_dict = dict(index=node_indices,
                         type=[random.choice(string.ascii_lowercase) for _ in self.nodes],
                         test_attr=[self.nodes[node]['test_attr'] for node in self.nodes]
                         )

        return node_dict, edge_dict
