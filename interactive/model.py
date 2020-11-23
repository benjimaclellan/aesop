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

    def export_as_dict(self, graph):
        ### Draw quadratic bezier paths
        def bezier(start, end, control1, control2, steps):
            return [(1 - s) ** 3 * start + 3 * (1 - s) ** 2 * s * control1 + 3 * (1 - s) * s ** 2 * control2 + s ** 3 * end for s in steps]

        # layout = self.get_layout(graph)
        layout = graph.optical_system_layout()

        xmin = min([xy[0] for xy in layout.values()])
        xmax = max([xy[0] for xy in layout.values()])
        ymin = min([xy[1] for xy in layout.values()])
        ymax = max([xy[1] for xy in layout.values()])
        for node, xy in layout.items():
            new_x = ((xy[0] - xmin)/(xmax - xmin) - 0.5) * 2.0
            new_y = ((xy[1] - ymin)/(ymax - ymin) - 0.5) * 2.0
            layout[node] = (new_x, new_y)

        xs, ys = [], []
        steps = [i / 100. for i in range(100)]
        for (u, v, k) in graph.edges:
            number_of_edges = graph.number_of_edges(u, v)
            xs.append(bezier(layout[u][0], layout[v][0],
                             (layout[u][0]+layout[v][0])/(number_of_edges + 1) * (1 + k),
                             (layout[u][0]+layout[v][0])/(number_of_edges + 1) * (1 + k),
                             steps))
            ys.append(bezier(layout[u][1], layout[v][1],
                             layout[u][1],
                             layout[v][1],
                             steps))


        edge_dict = dict(start=[edge[0] for edge in graph.edges],
                         end=[edge[1] for edge in graph.edges],
                         component=[graph.edges[edge]['model'].node_acronym for edge in graph.edges],
                         edge=[edge for edge in graph.edges],
                         xs=xs,
                         ys=ys
                         )

        node_dict = dict(index=[node for node in graph.nodes],
                         type=[random.choice(string.ascii_lowercase) for _ in graph.nodes],
                         component=[graph.nodes[node]['model'].node_acronym for node in graph.nodes]
                         )

        return layout, node_dict, edge_dict

    @staticmethod
    def get_layout(graph):
        graph_layout = nx.kamada_kawai_layout(graph)
        return graph_layout
