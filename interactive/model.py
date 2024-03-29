import dill
import networkx as nx
import random
import string
import sys
import pathlib
import platform
import os
import autograd.numpy as np
import json

# adds the ASOPE directory on any OS
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

from simulator.fiber.assets.functions import psd_, power_
from lib.minimal_save import build_from_minimal_graph_info


#%%
def load_one_minimal_graph(directory=None, file=None):
    filepath = pathlib.Path(directory).joinpath(file)
    with open(filepath, 'rb') as f:
        graph_json = json.load(f)
    return build_from_minimal_graph_info(graph_json)

# %% save various parts of
def load_single_object(directory=None, file=None, verbose=True):
    """

    :param directory: the relative data directory. should nto include a filename
    :param file: filename string
    :param verbose: boolean flag for print statements
    :return:
    """
    filepath_raw = pathlib.Path(directory).joinpath(file)
    # check if it is a reduced graph or full pickle, load accordingly
    if filepath_raw.suffix == '.json':
        # this is a reduced graph
        if verbose:
            print(f'Loading JSON object from {str(filepath_raw)}')
        with open(filepath_raw, 'rb') as f:
            json_data = json.load(f)
        obj_tmp = build_from_minimal_graph_info(json_data)

    elif filepath_raw.suffix == '.pkl':
        # it is a full pickled object
        if verbose:
            print(f'Loading pickle object from {str(filepath_raw)}')
        with open(filepath_raw, 'rb') as f:
            obj_tmp = dill.load(f)

    else:
        UserWarning('This is not a valid graph datatype. Please check the file path.', filepath_raw)

    return obj_tmp


class Model(object):
    """
    The Model class is the only one which interacts directly with ASOPE.
    It stores the graph, propagator, etc. objects.
    """
    verbose = False

    def __init__(self):
        self.graph = None
        self.propagator = None

        self.graph_edge_data = dict(start=[], end=[])  # these will hold all possible information from the edges
        self.graph_node_data = dict(index=[], type=[], test_attr=[])  # holds all possible information for each node, node

        self._graph_layout = {}

        self.graph_hessian_data = np.ones([1,1])

        self.table_edge_data = {}
        self.table_node_data = {}

        self.plot_edge_data = {}
        return

    def load_new_graph(self, folder, file):
        # loads a graph from a filename
        # graph_filename = pathlib.Path(folder).joinpath(file)
        # if self.verbose: print(f'Loading graph from {graph_filename}')
        # with open(graph_filename, 'rb') as file:
        #     full_graph = dill.load(file)


        # full_graph = load_single_object(directory=folder, file=file, verbose=True)
        full_graph = load_one_minimal_graph(directory=folder, file=file)

        graph = nx.convert_node_labels_to_integers(full_graph, first_label=0, ordering='default', label_attribute=None)

        model_type_counts = {}
        for node_edge in list(graph.nodes) + list(graph.edges):
            if type(node_edge) is tuple:
                model = graph.edges[node_edge]['model']
            else:
                model = graph.nodes[node_edge]['model']
            model_type = type(model)
            if model_type not in model_type_counts.keys():
                model_type_counts[model_type] = 0
            else:
                model_type_counts[model_type] += 1
            model.components_name_numbers = model.number_of_parameters * [f'{model.node_acronym}{model_type_counts[model_type]}']
            model.components_name_number = f'{model.node_acronym}{model_type_counts[model_type]}'

        prop_filepath = pathlib.Path(folder).joinpath('propagator.pkl')
        if self.verbose: print(f"Loading propagator from {prop_filepath}")
        with open(prop_filepath, 'rb') as file:
            propagator = dill.load(file)

        eval_filepath = pathlib.Path(folder).joinpath('evaluator.pkl')
        if self.verbose: print(f"Loading evaluator from {eval_filepath}")
        with open(eval_filepath, 'rb') as file:
            evaluator = dill.load(file)

        graph.initialize_func_grad_hess(propagator, evaluator)

        attr = graph.extract_attributes_to_list_experimental(attributes=['parameters'])

        # self.update_graph_hessian_data(graph)
        # graph.func(attr['parameters'])
        graph.propagate(propagator)

        self.graph = graph
        self.propagator = propagator

        self.update_graph_layout(graph)

        self.get_graph_edge_data(graph)
        self.get_graph_node_data(graph)

        self.get_table_edge_data(graph)
        self.get_plot_edge_data(graph, propagator)
        if self.verbose: print('Successfully loaded and simulated the graph')
        return

    """
    Gets the data which used in the interactive graph renderer. 
    Two data sources are used: one for the edges and one for the nodes, both are given to the Control as a dictionary.
    Some dictionary keys are required by Bokeh, e.g. 'start', 'end', 'index'
    """
    def get_graph_edge_data(self, graph):

        xs, ys = self.get_optical_system_edge_curves(graph)

        graph_edge_data = dict(index=[i for i in range(len(graph.edges))],
                               start=[edge[0] for edge in graph.edges],
                               end=[edge[1] for edge in graph.edges],
                               component=[graph.edges[edge]['model'].components_name_number for edge in graph.edges],
                               edge=[edge for edge in graph.edges],
                               xs=xs,
                               ys=ys,
                               )

        self.graph_edge_data = graph_edge_data
        return

    def get_graph_node_data(self, graph):
        graph_node_data = dict(index=[node for node in graph.nodes],
                               type=[random.choice(string.ascii_lowercase) for _ in graph.nodes],
                               component=[graph.nodes[node]['model'].components_name_number for node in graph.nodes]
                               )
        self.graph_node_data = graph_node_data
        return

    """
    Get the information that will be displayed in the data table
    """
    def get_table_edge_data(self, graph):
        table_edge_data = {}
        for i, edge in enumerate(graph.edges):
            edge_data = {}

            component = graph.edges[edge]['model']
            edge_data['parameters'] = component.parameters  # TODO: only getting unlock parameters here, but not elsewhere
            edge_data['parameter_names'] = component.parameter_names
            edge_data['model'] = component.components_name_numbers

            table_edge_data[i] = edge_data

        self.table_edge_data = table_edge_data
        return

    def get_table_node_data(self, graph):
        raise NotImplementedError('Not using the nodes in the table yet.')

    def get_plot_edge_data(self, graph, propagator):
        """
        Data for use in the interactive plots is stores as a nested dictionary
            plot_edge_data = {plot_id : {edge: [list of data to be plotted]}
        We use a similar data structure in the Model, but slightly different
        """

        plot_edge_data = {'prop_time': {}, 'prop_freq': {}, 'tran_time': {}, 'tran_freq': {}}
        for i, edge in enumerate(graph.edges):
            state = np.squeeze(self.graph.measure_propagator(edge))
            # print(f'State to plot is type {type(state)}')
            plot_edge_data['prop_time'][i] = [dict(x=propagator.t, y=power_(state).astype('float'))]
            plot_edge_data['prop_freq'][i] = [dict(x=propagator.f, y=np.log10(psd_(state, dt=propagator.dt, df=propagator.df).astype('float')))]

            plot_edge_data['tran_time'][i], plot_edge_data['tran_freq'][i] = [], []
            if graph.edges[edge]['model'].transform is not None:
                for j, (dof, transform, label) in enumerate(graph.edges[edge]['model'].transform):
                    if dof == 't':
                        plot_edge_data['tran_time'][i].append(dict(x=propagator.t, y=transform.astype('float')))
                    elif dof == 'f':
                        plot_edge_data['tran_freq'][i].append(dict(x=propagator.f, y=transform.astype('float')))

        self.plot_edge_data = plot_edge_data

    def update_graph_layout(self, graph):
        self._graph_layout = graph.optical_system_layout()
        return

    def get_optical_system_edge_curves(self, graph):

        # draw cubic bezier paths
        def bezier(start, end, control1, control2, steps):
            return [(1 - s) ** 3 * start + 3 * (1 - s) ** 2 * s * control1 + 3 * (1 - s) * s ** 2 * control2 + s ** 3 * end for s in steps]

        layout = self._graph_layout

        xmin = min([xy[0] for xy in layout.values()])
        xmax = max([xy[0] for xy in layout.values()])
        ymin = min([xy[1] for xy in layout.values()])
        ymax = max([xy[1] for xy in layout.values()])
        for node, xy in layout.items():
            new_x = ((xy[0] - xmin) / (xmax - xmin) - 0.5) * 2.0
            new_y = ((xy[1] - ymin) / (ymax - ymin) - 0.5) * 2.0
            layout[node] = (new_x, new_y)

        xs, ys = [], []
        steps = [i / 100. for i in range(100)]
        for (u, v, k) in graph.edges:
            number_of_edges = graph.number_of_edges(u, v)
            xs.append(bezier(layout[u][0], layout[v][0],
                             (layout[u][0] + (layout[v][0] - layout[u][0]) / (number_of_edges + 1) * (1 + k)),
                             (layout[u][0] + (layout[v][0] - layout[u][0]) / (number_of_edges + 1) * (1 + k)),
                             steps))
            ys.append(bezier(layout[u][1], layout[v][1],
                             layout[u][1],
                             layout[v][1],
                             steps))
        return xs, ys

    def update_graph_hessian_data(self, graph):
        attr = graph.extract_attributes_to_list_experimental(attributes=['parameters', 'parameter_names', 'components_name_numbers'])

        parameter_text = [f'{p}-{c}' for (p, c) in zip(attr['parameter_names'], attr['components_name_numbers'])]
        names_i, names_j = np.meshgrid(np.array(parameter_text),
                                       np.array(parameter_text))
        self.graph_hessian_data = dict(hess=graph.hess(attr['parameters']),
                                       x_name=names_i,
                                       y_name=names_j)
        return

    @property
    def graph_layout(self):
        return self._graph_layout
