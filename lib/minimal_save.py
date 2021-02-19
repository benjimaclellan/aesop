#%%

import sys
sys.path.append('..')

import collections
import json

import config.config as configuration

from lib.graph import Graph


# flatten a nested dictionary, will be used to flatten dictionary of model-name:model-object information in config
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_minimal_graph_info(graph):
    node_models = {}
    node_models_params = {}
    for i, node in enumerate(graph.nodes):
        model_name = join_model_nickname(graph.nodes[node]['model'], i)
        node_models[node] = model_name
        node_models_params[model_name] = graph.nodes[node]['model'].parameters

    edge_models = {}
    edge_models_adj = {}
    edge_models_params = {}
    for i, edge in enumerate(graph.edges, i):
        model_name = join_model_nickname(graph.edges[edge]['model'], i)
        new_edge = (node_models[edge[0]], node_models[edge[1]], edge[2])
        edge_models[new_edge] = model_name
        edge_models_adj[model_name] = new_edge
        edge_models_params[model_name] = graph.edges[edge]['model'].parameters

    graph_info = dict(edge_models_adj=edge_models_adj,
                      node_models_params=node_models_params,
                      edge_models_params=edge_models_params)

    json_data = {'graph_info': graph_info}
    return json_data


def build_from_minimal_graph_info(json_data):

    lookup = flatten(configuration.NODE_TYPES_ALL)
    #
    # with open(filepath, 'r') as f:
    #     json_data = json.load(f)

    graph_info = json_data['graph_info']
    edge_models_adj = graph_info['edge_models_adj']
    node_models_params = graph_info['node_models_params']
    edge_models_params = graph_info['edge_models_params']

    nodes = {node_model: lookup[split_model_nickname(node_model)[0]]() for node_model in node_models_params.keys()}
    edges = {tuple(edge_tuple): lookup[split_model_nickname(edge_model)[0]]() for edge_model, edge_tuple in edge_models_adj.items()}

    new_graph = Graph.init_graph(nodes, edges)
    new_graph.update_graph()

    for node in new_graph.nodes:
        params = node_models_params[node]
        if new_graph.nodes[node]['model'].number_of_parameters != len(params):
            raise ValueError("the number of parameters for minimal saved information does match for node {}".format(node))
        new_graph.nodes[node]['model'].parameters = params

    tmp = {tuple(val) : key for key, val in edge_models_adj.items()}
    for edge in new_graph.edges:
        model_name = tmp[edge]
        params = edge_models_params[model_name]
        if new_graph.edges[edge]['model'].number_of_parameters != len(params):
            raise ValueError("the number of parameters for minimal saved information does match for edge {}".format(edge))
        new_graph.edges[edge]['model'].parameters = params
    return new_graph


def join_model_nickname(model, idint):
    return str(model.__class__.__name__) + ".{:03.0f}".format(idint)


def split_model_nickname(nickname):
    return nickname.split('.')