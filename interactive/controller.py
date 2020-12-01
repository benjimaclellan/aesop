import time
import networkx as nx


class Controller(object):

    """
    The Control class is an intermediary between the View and Model classes.
    """

    verbose = False

    def __init__(self, model):
        self.model = model

        self._graph_layout = {}
        self._graph_edge_data = dict(start=[], end=[])
        self._graph_node_data = dict(index=[], type=[], test_attr=[])

        self._table_data = dict(parameter_names=[], parameters=[], model=[])
        return

    def load_new_graph(self, filepath):
        if self.verbose: print('Loading new graph')
        if type(filepath) is not str:
            raise TypeError('Cannot load graph, filepath must be a string')
        self.model.load_new_graph(filepath)
        return

    def update_table_data(self, new_indices):
        table_data = {column: [] for column in ['parameters', 'parameter_names', 'model']}
        for index in new_indices:
            for column in table_data.keys():
                table_data[column] += self.model.table_edge_data[index][column]
        self._table_data = table_data
        return

    @property
    def plot_edge_data(self):
        return self.model.plot_edge_data

    """
    Properties of data which will be accessed by the View object
    """
    @property
    def table_data(self):
        return self._table_data

    @property
    def graph_layout(self):
        if self.model.graph is not None:
            self._graph_layout = self.model.graph_layout
        else:
            if self.verbose: print('No graph is loaded yet')
            self._graph_layout = {}
        return self._graph_layout

    @property
    def graph_edge_data(self):
        return self.model.graph_edge_data

    @property
    def graph_node_data(self):
        return self.model.graph_node_data


