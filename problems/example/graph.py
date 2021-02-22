"""

"""
print("The Graph class in problems.example.graph is deprectated. Please only use this for importing legacy Pickle objects")


import autograd.numpy as np
from autograd import grad, hessian, jacobian, elementwise_grad

import networkx as nx
import copy
import matplotlib.pyplot as plt
import matplotlib.cbook as cb

from itertools import cycle
import warnings

from uuid import uuid4
from collections import namedtuple

from lib.base_classes import Graph as GraphParent
from problems.example.assets.functions import power_, psd_
from problems.example.assets.additive_noise import AdditiveNoise
from lib.functions import scale_units
from lib.autodiff_helpers import unwrap_arraybox_list

from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink


class Graph(GraphParent):
    """Parent class
    """

    __internal_var = None

    @classmethod
    def init_graph(cls, nodes: dict, edges: dict):
        graph = cls()
        for node, model in nodes.items():
            graph.add_node(node, **{'model': model, 'name': model.__class__.__name__, 'lock': False})
        for edge, model in edges.items():
            graph.add_edge(edge[0], edge[1], **{'model': model, 'name': model.__class__.__name__, 'lock': False})

        sources = [node for node in graph.nodes if graph.get_in_degree(node) == 0]
        sinks = [node for node in graph.nodes if graph.get_out_degree(node) == 0]
        for source in sources: graph.nodes[source]['source-sink'] = 'source'
        for sink in sinks: graph.nodes[sink]['source-sink'] = 'sink'

        graph._propagation_order = None
        graph._propagator_saves = {}
        graph.update_graph()

        # initialize description needed for speciation
        graph.speciation_descriptor = None

        # initialize evolution probability matrix
        graph.evo_probabilities_matrix = None
        graph.last_evo_record = None

        # initialize variables to store function handles for grad & hess
        graph.func = None
        graph.grad = None
        graph.hess = None

        graph.scaled_hess_matrix = None
        # Attribute: self.scaled_hess

        graph.current_uuid = uuid4()
        graph.parent_uuid = uuid4()

        graph.score = None
        return graph

    @classmethod
    def duplicate_and_simplify_graph(cls, graph):
        # copy nodes and edges - used before saving to avoid potential recursion errors in serialization
        nodes, edges = {}, {}
        for node in graph.nodes:
            nodes[node] = copy.deepcopy(graph.nodes[node]['model'])
        for edge in graph.edges:
            edges[edge] = copy.deepcopy(graph.edges[edge]['model'])
        graph_copy = cls.init_graph(nodes=nodes, edges=edges)
        return graph_copy

    # def __init__(self):
    #     """
    #     """
    #     super().__init__()

    def update_graph(self):
        propagation_order = list(nx.topological_sort(self))
        self._propagation_order = propagation_order

        # tell node models how many incoming and outgoing nodes there are
        for node in self.nodes:
            self.nodes[node]['model'].update_attributes(self.get_in_degree(node), self.get_out_degree(node))
        return

    def clean_graph(self):
        self.func = None
        self.grad = None
        self.hess = None
        self.scaled_hess_matrix = None
        self.clear_propagation()

    def __str__(self):
        str_rep = ''
        for node in self.nodes:
            str_rep += f"{node}: {self.nodes[node]['model'].__class__.__name__}\n"
        for edge in self.edges:
            str_rep += f"{edge}: {self.edges[edge]['model'].__class__.__name__}\n"
        return str_rep

    @property
    def interfaces(self):
        Interface = namedtuple('Interface', 'node edge')
        interfaces = [Interface(node=edge[0], edge=edge) for edge in self.edges if self.in_degree[edge[0]] != 0] + \
                     [Interface(node=edge[1], edge=edge) for edge in self.edges if self.out_degree[edge[1]] != 0]
        return interfaces

    def function_wrapper(self, propagator, evaluator, exclude_locked=True):
        """ returns a function handle that accepts only parameters and returns the score. used to initialize the hessian analysis """

        def _function(_parameters, _graph, _propagator, _evaluator, _models, _parameter_index):
            _graph.distribute_parameters_from_list(_parameters, _models, _parameter_index)
            _graph.propagate(_propagator)
            score = _evaluator.evaluate_graph(_graph, _propagator)
            return score

        info = self.extract_attributes_to_list_experimental([], get_location_indices=True,
                                                            exclude_locked=exclude_locked)

        func = lambda parameters: _function(parameters, self, propagator, evaluator, info['models'],
                                            info['parameter_index'])
        return func

    def initialize_func_grad_hess(self, propagator, evaluator, exclude_locked=True):
        self.func = self.function_wrapper(propagator, evaluator, exclude_locked=exclude_locked)
        self.grad = grad(self.func)
        hess_tmp = hessian(self.func)  # hessian requires a numpy array, so wrap in this way
        self.hess = lambda parameters: hess_tmp(np.array(parameters))

    @property
    def scaled_hess(self):
        attributes = self.extract_attributes_to_list_experimental(['parameter_imprecisions'])
        parameter_imprecisions = np.expand_dims(np.array(attributes['parameter_imprecisions']), axis=1)

        scale_matrix = np.matmul(parameter_imprecisions, parameter_imprecisions.T)

        return lambda parameters: self.hess(np.array(parameters)) * scale_matrix

    def get_in_degree(self, node):
        return len(self.get_in_edges(node))

    def get_out_degree(self, node):
        return len(self.get_out_edges(node))

    def get_in_edges(self, node):
        """ """
        return [(u, v, k) for (u, v, k) in self.edges if v == node]

    def get_out_edges(self, node):
        """ """
        return [(u, v, k) for (u, v, k) in self.edges if u == node]

    def suc(self, node):
        """Return the successors of a node (nodes which follow the current one) as a list
        """
        return list(self.successors(node))

    def pre(self, node):
        """Return the predeccessors of a node (nodes which lead to the current one) as a list
        """
        return list(self.predecessors(node))

    def get_next_valid_node_ID(self):
        return uuid4()

    def clear_propagation(self):
        self._propagator_saves = {}  # maybe this fixes weird, unphysical results from systems

    def get_output_signal(self, propagator, node=None, save_transforms=False):
        """
        Propagates (with noise) through the graph, and returns the signal at node (if node is None, output node picked)
        Does not change value of AdditiveNoise.simulate_with_noise (though it's not threadsafe: it's changed within the method)

        :param propagator: propagator with which to evaluate the signal
        :param node: node to evaluate. If None, output node is used
        :save_transforms: True to save transforms for visualisation. False otherwise

        :returns: noisy signal at node
        """
        # get noisy output signal (or noiseless! Whatever it's previously set at (but usually noisy))
        if (node is None):
            node = self.get_output_node()

        start_noise_status = AdditiveNoise.simulate_with_noise
        AdditiveNoise.simulate_with_noise = True

        self.propagate(propagator, save_transforms=save_transforms)

        AdditiveNoise.simulate_with_noise = start_noise_status  # restore initial value
        return self.measure_propagator(node)

    def get_output_signal_pure(self, propagator, node=None, save_transforms=False):
        """
        Propagates (without noise) through the graph, and returns the signal at node (if node is None, output node picked)
        Does not change value of AdditiveNoise.simulate_with_noise (though it's not threadsafe: it's changed within the method)

        :param propagator: propagator with which to evaluate the signal
        :param node: node to evaluate. If None, output node is used
        :save_transforms: True to save transforms for visualisation. False otherwise

        :returns: noiseless signal at node
        """
        if (node is None):
            node = self.get_output_node()

        start_noise_status = AdditiveNoise.simulate_with_noise
        AdditiveNoise.simulate_with_noise = False

        self.propagate(propagator, save_transforms=save_transforms)

        AdditiveNoise.simulate_with_noise = start_noise_status  # restore initial value
        return self.measure_propagator(node)

    def get_output_noise(self, propagator, node=None, save_transforms=False):
        """
        Returns the noise at node (if node is None, output node picked)
        Does not change value of AdditiveNoise.simulate_with_noise (though it's not threadsafe: it's changed within the method)

        :param propagator: propagator with which to evaluate the signal
        :param node: node to evaluate. If None, output node is used
        :save_transforms: True to save transforms for visualisation. False otherwise

        :returns: signal at node
        """
        noisy = self.get_output_signal(propagator, node=node, save_transforms=save_transforms)
        noiseless = self.get_output_signal_pure(propagator, node=node, save_transforms=save_transforms)

        return noisy - noiseless

    def resample_all_noise(self, seed=None):
        """
        Resamples noise for all noise models in the graph

        :param seed: seed to use prior to resampling
        """
        if (seed is not None):
            np.random.seed(seed)

        for node in self.nodes:
            noise_model = self.nodes[node]['model'].noise_model
            if (noise_model is not None):
                noise_model.resample_noise()

        for edge in self.edges:
            if self._propagate_on_edges and 'model' in self.edges[
                edge]:  # this also simulates components stored on the edges, if there is a model on that edge
                noise_model = self.edges[edge]['model'].noise_model
                if (noise_model is not None):
                    noise_model.resample_noise()

    def display_noise_contributions(self, propagator, node=None, title=''):
        noisy = self.get_output_signal(propagator, node=node)
        noiseless = self.get_output_signal_pure(propagator, node=node)
        noise = self.get_output_noise(propagator, node=node)
        fig, ax = plt.subplots(2, 1)
        linestyles = ['-', ':', ':']
        labels = ['noisy', 'noiseless', 'noise']
        noisy_max = np.max(psd_(noisy, propagator.dt, propagator.df))
        for (state, line, label) in zip([noisy, noiseless, noise], linestyles, labels):
            ax[0].plot(propagator.t, power_(state), ls=line, label=label)
            ax[0].set_xlabel('time (s)')
            ax[0].set_ylabel('power (W)')
            psd = psd_(state, propagator.dt, propagator.df)
            ax[1].plot(propagator.f, 10 * np.log10(psd / noisy_max), ls=line, label=label)
            ax[1].set_xlabel('frequency (Hz)')
            ax[1].set_ylabel('Normalized PSD (dB)')

        ax[0].legend()
        ax[1].legend()
        plt.title(title)
        plt.show()

    def get_output_node(self):
        """
        Returns the output node (i.e. the node with no output vertices). If there are multiple output nodes, returns the first
        """
        return [node for node in self.nodes if not self.out_edges(node)][0]  # finds node with no outgoing edges

    def propagate(self, propagator, save_transforms=False):
        """"
        Uses no deepcopy functionality - hopefully avoiding issues with autograd.
        Instead, the propagator object at each nodes is saved to a separate dictionary, where the key is the node
        """

        # raise RuntimeError('this has not be re-implemented yet')

        self._propagator_saves = {}  # will save each propagator here
        propagation_order = nx.topological_sort(self)
        # propagation_order = self.propagation_order

        for node in propagation_order:
            # get the incoming optical field
            if not self.pre(node):
                tmp_states = [propagator.state]
            else:  # if we have incoming nodes to get the propagator from
                tmp_states = []  # initialize list to add all incoming propagators to
                for incoming_edge in self.get_in_edges(node):  # loop through incoming edges
                    tmp_states.append(self._propagator_saves[incoming_edge])

            states = self.nodes[node]['model'].propagate(tmp_states, propagator, self.in_degree(node),
                                                         self.out_degree(node), save_transforms=save_transforms)

            if self.get_out_degree(node) == 0:
                self._propagator_saves[node] = states[0]  # save the final state at the source node

            for edge_index, outgoing_edge in enumerate(self.get_out_edges(node)):
                signal = self.edges[outgoing_edge]['model'].propagate(states[edge_index], propagator,
                                                                      save_transforms=save_transforms)

                # add noise to propagation
                noise_model = self.edges[outgoing_edge]['model'].noise_model
                if noise_model is not None:
                    signal = noise_model.add_noise_to_propagation(signal, propagator)

                self._propagator_saves[
                    outgoing_edge] = signal  # we can use the edge as a hashable key because it is immutable (so we can use tuples, but not lists)
        return self

    def measure_propagator(self, node_edge):
        return self._propagator_saves[node_edge]

    def visualize_transforms_dof(self, ax, propagator, dof='f', label_verbose=1):
        for node in self.nodes():
            if self.nodes[node]['model'].transform is not None:
                for _, (dof_i, transform, label) in enumerate(self.nodes[node]['model'].transform):
                    if label_verbose == 0:
                        label_str = '{}'.format(label)
                    else:
                        label_str = 'Node {} | {} | {}'.format(node, self.nodes[node]['name'], label)

                    if (dof_i == dof) and (dof == 't'):
                        ax.plot(propagator.t, transform / np.max(transform), label=label_str)
                    elif (dof_i == dof) and (dof == 'f'):
                        ax.plot(propagator.f, transform / np.max(transform), label=label_str)
            ax.legend()

    def visualize_transforms(self, nodes_to_visualize, propagator):
        """
        each node's propagate function (if applicable) can save the transformations (as functions of t or f) into the class variable
        node.transform, where node.transform = (dof, transfer function, text label) where dof is t or f

        :param nodes_to_visualize: list of nodes to look for transfer functions to plot
        :param propagator: same propagator as in the simulations
        :return:
        """
        fig, ax = plt.subplots(2, 1)
        for node in nodes_to_visualize:
            if self.nodes[node]['model'].transform is not None:
                for _, (dof, transform, label) in enumerate(self.nodes[node]['model'].transform):
                    label_str = 'Node {} | {} | {}'.format(node, self.nodes[node]['name'], label)
                    if dof == 't':
                        ax[0].plot(propagator.t, transform / np.max(transform), label=label_str)
                    elif dof == 'f':
                        ax[1].plot(propagator.f, transform / np.max(transform), label=label_str)
        for axi in ax:
            axi.legend()
        ax[0].set_xlabel('Time')
        ax[1].set_xlabel('Frequency')

        scale_units(ax[0], unit='s', axes=['x'])
        scale_units(ax[1], unit='Hz', axes=['x'])

        plt.show()
        return

    def optical_system_layout(self, debug=False):
        pos = {}
        order = list(nx.topological_sort(self))
        current_row = set([order[0]])
        nodes_remaining = set(order)
        rows = {}

        flag = True
        row_i, col_i = 0, 0
        while flag:
            # If there are no more nodes to find positions for, break out and continue plotting
            if len(nodes_remaining) == 0:
                flag = False
                break

            # add all of the successors of the current row into a set. we will remove some of these in the next step
            # each row is spaced evenly in the horizontal direction
            next_row = set()
            for i, node_i in enumerate(list(current_row)):
                pos[node_i] = (row_i, np.random.rand() * 4)
                next_row.update(set(self.suc(node_i)))

            # we will remove some nodes that depend on other nodes further down the DAG. this is based on ancestry
            nodes_to_remove = set()
            for node_i in next_row:
                for node_j in next_row:
                    ancestors = nx.algorithms.ancestors(self, node_i)
                    if debug:
                        print(f'current_row:{current_row}, next_row:{next_row}, node_i:{node_i}, node_j:{node_j}, '
                              f'nodes_to_remove:{nodes_to_remove}, ancestors:{ancestors}, '
                              f'nodes_remainingL {nodes_remaining}')
                    # if node_i has any other ancestors, we won't plot it in this row
                    if node_j in ancestors:
                        nodes_to_remove.update(set([node_i]))
            next_row -= nodes_to_remove

            row_i += 1
            nodes_remaining -= current_row
            current_row = next_row
        return pos

    def draw(self, ax=None, labels=None, method='grid', ignore_warnings=True, debug=False):
        if ignore_warnings: warnings.simplefilter('ignore', category=(FutureWarning, cb.mplDeprecation))

        # pos = nx.spring_layout(self)
        pos = self.optical_system_layout(debug=debug)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        nx.draw_networkx_nodes(self, pos, ax=ax, node_color='r',
                               label=[self.nodes[node]['model'].node_acronym for node in self.nodes])
        for n, p in pos.items():
            ax.annotate(self.nodes[n]['model'].node_acronym,
                        xy=p, xycoords='data',
                        xytext=p, textcoords='data',
                        va='center', ha='center'
                        )

        def _arrow_center(pos_start, pos_end, rad):
            """
            Radius sign determines whether the arrow goes clockwise (neg radius) or counterclockwise (pos radius)
            See: https://matplotlib.org/3.1.1/gallery/userdemo/connectionstyle_demo.html
            """
            pos_start, pos_end = np.asarray(pos_start), np.asarray(pos_end)

            # 1. Find centre of circle
            dist = np.sqrt(np.sum(np.power(pos_end - pos_start, 2)))
            pos_avg = pos_start / 2 + pos_end / 2

            # 2. Find angle of the line perpendicular to the line between start and end
            theta = np.arctan2(pos_end[1] - pos_start[1], pos_end[0] - pos_start[0]) - np.pi / 2
            perp_vector = rad * dist * np.array([np.cos(theta), np.sin(theta)])

            # 3. New centre
            arrow_centre = pos_avg + perp_vector * 0.7
            return (arrow_centre[0], arrow_centre[1])

        for e in self.edges:
            rad = 0.4 * e[2]
            ax.annotate('',
                        xy=pos[e[1]], xycoords='data',
                        xytext=pos[e[0]], textcoords='data',
                        arrowprops=dict(arrowstyle="-|>", color="0.1",
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr', str(rad)),
                                        ),
                        )
            arrow_centre = _arrow_center(pos[e[0]], pos[e[1]], rad)
            ax.annotate(self.edges[e]['model'].node_acronym,
                        xy=arrow_centre, xycoords='data',
                        )
        ax.set_aspect('equal')
        plt.axis('off')
        if ignore_warnings: warnings.simplefilter('always', category=(FutureWarning, cb.mplDeprecation))
        return ax

    @property
    def propagation_order(self):
        """Returns the sorted order of nodes (based on which reverse walking the graph)
        """
        return self._propagation_order

    def assert_number_of_edges(self):
        """Loops through all nodes and checks that the proper number of input/output edges are connected
        """
        assert nx.is_directed_acyclic_graph(self)
        return

    def sample_parameters(self, probability_dist='uniform', **kwargs):
        """ Samples new parameters for each node-type """
        attributes = ['parameters', 'lower_bounds', 'upper_bounds', 'data_types', 'step_sizes']
        model_attributes = self.extract_attributes_to_list_experimental(attributes, get_location_indices=True)
        parameters = self.sample_parameters_callable(model_attributes['parameters'], model_attributes['lower_bounds'],
                                                     model_attributes['upper_bounds'], model_attributes['data_types'],
                                                     model_attributes['step_sizes'], probability_dist=probability_dist,
                                                     **kwargs)

        self.distribute_parameters_from_list(parameters,
                                             model_attributes['models'],
                                             model_attributes['parameter_index'])
        return

    def sample_parameters_to_list(self, probability_dist='uniform', **kwargs):
        """ Samples new parameters for each node-type """

        # TODO Change this as well
        attributes = ['parameters', 'lower_bounds', 'upper_bounds', 'data_types', 'step_sizes']
        model_attributes = self.extract_attributes_to_list_experimental(attributes, get_location_indices=False)
        parameters = self.sample_parameters_callable(model_attributes['parameters'], model_attributes['lower_bounds'],
                                                     model_attributes['upper_bounds'], model_attributes['data_types'],
                                                     model_attributes['step_sizes'], probability_dist=probability_dist,
                                                     **kwargs)

        return parameters

    def get_parameter_bounds(self):
        """
        Returns the lower bounds, upper bounds of parameters, in the order returned by all other functions

        :Return: <lower bounds (array)>, <upper bounds (array)>
        """
        attributes = ['lower_bounds', 'upper_bounds']
        model_attributes = self.extract_attributes_to_list_experimental(attributes, get_location_indices=False)

        return np.array(model_attributes['lower_bounds']), np.array(model_attributes['upper_bounds'])

    def get_models(self, include_node_edge=False):
        if include_node_edge:
            return [(self.nodes[node]['model'], node) for node in self.nodes] + [(self.edges[edge]['model'], edge) for
                                                                                 edge in self.edges]
        return [[self.nodes[node]['model']] for node in self.nodes] + [[self.edges[edge]['model']] for edge in
                                                                       self.edges]

    def extract_attributes_to_list_experimental(self, attributes, get_location_indices=True, get_node_edge_index=False,
                                                exclude_locked=True):
        """ experimental: to extract model variables to a list, based on a list of variables names """

        # first generate multiple empty lists for each attribute of attributes
        model_attributes = {}
        for attribute in attributes:
            model_attributes[attribute] = []
        if get_location_indices:
            model_attributes['models'], model_attributes['parameter_index'] = [], []
        if get_node_edge_index:
            model_attributes['node_edge_index'] = []
            # these will help translate from a list of parameters/parameter info to a graph structure

        models = self.get_models(include_node_edge=get_node_edge_index)
        for model in models:
            if not model[0].node_lock:
                for i, lock in enumerate(model[0].parameter_locks):
                    if not lock:
                        for attribute in attributes:
                            model_attributes[attribute].append(getattr(model[0], attribute)[i])
                        if get_location_indices:
                            model_attributes['models'].append(model[0])
                            model_attributes['parameter_index'].append(i)
                        if get_node_edge_index:
                            model_attributes['node_edge_index'].append(model[1])
        return model_attributes

    def extract_parameters_to_list(self, exclude_locked=True):
        """ Extracts the current parameters, bounds and information for re-distributing from the graph structure """

        attributes = ['parameters', 'lower_bounds', 'upper_bounds']

        model_attributes = self.extract_attributes_to_list_experimental(attributes,
                                                                        get_location_indices=True,
                                                                        exclude_locked=exclude_locked)

        parameters = model_attributes['parameters']
        models = model_attributes['models']
        parameter_index = model_attributes['parameter_index']
        lower_bounds = model_attributes['lower_bounds']
        upper_bounds = model_attributes['upper_bounds']

        return unwrap_arraybox_list(parameters), models, parameter_index, lower_bounds, upper_bounds

    def distribute_parameters_from_list(self, parameters, models, parameter_index):
        """ from the lists created in 'extract_parameters_to_list', we distribute these (or altered versions, like in scipy.optimize.minimize) back to the graph"""
        for i, (parameter, model, parameter_ind) in enumerate(zip(parameters, models, parameter_index)):
            model.parameters[parameter_ind] = parameter
        return

    def inspect_parameters(self):
        """ Loops through all nodes & edge (if enabled) and prints information about the parameters """
        for model in self.get_models():
            model.inspect_parameters()

    def inspect_state(self, propagator, freq_log_scale=False, title=''):
        """ we loop through all nodes and plot the optical state *after* the node"""
        fig, ax = plt.subplots(2, 1)
        linestyles = cycle(['-', '--', '-.', ':'])

        # please note that we do not include the edges here (most of the time I doubt we will use edges, but it may be useful in the future)
        # for cnt, node in enumerate(self.propagation_order):
        for cnt, node in enumerate(reversed(self.propagation_order)):
            state = self.measure_propagator(node)
            line = {'ls': next(linestyles), 'lw': 3}
            ax[0].plot(propagator.t, power_(state), label=self.nodes[node]['model'].__class__.__name__, **line)
            psd = psd_(state, propagator.dt, propagator.df)
            if freq_log_scale:
                ax[1].plot(propagator.f, 10 * np.log10(psd / np.max(psd)), **line)
            else:
                ax[1].plot(propagator.f, 0.1 * cnt + psd / np.max(psd), **line)
        ax[0].legend()
        plt.title(title)
        plt.show()

    @staticmethod
    def sample_parameters_callable(parameters_current, lower_bounds, upper_bounds, data_types, step_sizes,
                                   probability_dist='uniform', **kwargs):
        """ Samples the new parameters from a given distribution and parameter bounds """
        parameter_details = zip(parameters_current, lower_bounds, upper_bounds, data_types, step_sizes)
        parameters = []
        for ind, (parameter_current, low, up, data_type, step) in enumerate(parameter_details):
            if probability_dist == 'uniform':
                parameter = uniform_sample(low=low, up=up, step=step, data_type=data_type)
            elif probability_dist == 'triangle':
                interval_width = kwargs['triangle_width'] if hasattr(kwargs, 'triangle_width') else 0.05
                parameter = triangle_sample(parameter=parameter_current, low=low, up=up,
                                            step=step, data_type=data_type, interval_width=interval_width)
            else:
                warnings.warn("This is not a valid sampling function, reverting to uniform")
                parameter = uniform_sample(low=low, up=up, step=step, data_type=data_type)
            parameters.append(parameter)
        return parameters

    def get_graph_info(self):
        nodes = list(self.nodes)
        edges = list(self.edges)
        models = [self.nodes[node]['model'].node_acronym for node in self.nodes]
        model_edges = [(self.nodes[i]['model'].node_acronym, self.nodes[j]['model'].node_acronym) for (i, j, _) in
                       self.edges]
        return nodes, edges, models, model_edges


# %% Sampling functions for mutation operations on parameters
def uniform_sample(low=0.0, up=1.0, step=None, data_type='int'):
    if data_type == 'float':
        if step is None:
            parameter = np.random.uniform(low, up)
        else:
            parameter = round(np.random.uniform(low, up) / step) * step
    elif data_type == 'int':
        if step is None:
            parameter = np.random.randint(low, up)
        else:
            parameter = np.round(np.random.randint(low / step, up / step)) * step
    else:
        raise ValueError('Unknown datatype in the current parameter')
    return parameter


def triangle_sample(parameter=None, low=0.0, up=1.0, step=None, data_type='int', triangle_width=0.05):
    """
    Experimental mutation operator, using triangular probability distribution
    This is od-hoc - one foreseeable issue is that the probability of drawing up | low is always 0
    """

    if parameter is None:
        raise RuntimeError("Current parameter must be passed to apply a Gaussian mutation")

    radius = triangle_width * (up - low) / 2
    left = parameter - radius if parameter - radius > low else low
    right = parameter + radius if parameter + radius < up else up
    parameter_old = float(parameter)
    if data_type == 'float':
        if step is None:
            parameter = np.random.triangular(left, parameter, right)
        else:
            parameter = round(np.random.triangular(left, parameter, right) / step) * step
    elif data_type == 'int':
        parameter = int(np.random.triangular(left, parameter, right))
    else:
        raise ValueError('Unknown datatype for this parameter')

    print('parameter_old {} | parameter {} | radius {} | left {} | right {}'.format(parameter_old, parameter, radius,
                                                                                    left, right))

    return parameter
