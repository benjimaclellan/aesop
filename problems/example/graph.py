"""

"""

import autograd.numpy as np
from autograd import grad, hessian, jacobian, elementwise_grad

import networkx as nx
import copy
import matplotlib.pyplot as plt
import matplotlib.cbook as cb

from itertools import cycle
import warnings

from lib.base_classes import Graph as GraphParent
from .assets.functions import power_, psd_
from .assets.additive_noise import AdditiveNoise
from lib.functions import scale_units


class Graph(GraphParent):
    """Parent class
    """

    __internal_var = None

    def __init__(self, nodes = dict(), edges = list(), propagate_on_edges = False, coupling_efficiency=1):
        """
        """
        super().__init__()

        for node, model in nodes.items():
            self.add_node(node, **{'model': model, 'name': model.__class__.__name__, 'lock': False})

        # add the models to the edges if propagating on edges (fibers, paths etc)
        #TODO: this maybe could be improved - its a little hacky (but it works for now, so leave for now)

        for edge in edges:
            if len(edge) == 2:
                self.add_edge(edge[0], edge[1])
            elif len(edge) > 2:
                self.add_edge(edge[0], edge[1], **{'model': edge[2], 'name': "FUNCTIONALITY NOT IMPLEMENTED", 'lock': False, 'states':None})
            else:
                raise TypeError("Incorrect number of arguments in the {} edge connection tuple".format(edge))
        self._propagate_on_edges = propagate_on_edges

        self._propagation_order = None

        self._propagator_saves = {}

        self.coupling_efficiency = coupling_efficiency
        if coupling_efficiency < 0 or coupling_efficiency > 1:
            raise ValueError(f'Coupling efficiency: {coupling_efficiency} is unphysical (0 <= efficiency <= 1)')
        
        # initialize description needed for speciation
        self.speciation_descriptor = None

        # initialize evolution probability matrix
        self.evo_probabilities_matrix = None
        self.last_evo_record = None

        # initialize variables to store function handles for grad & hess
        self.func = None
        self.grad = None
        self.hess = None

        self.score = None
    
    # def __str__(self):
    #     string_rep = ""
    #     for i in self.nodes:
    #         string_rep += f"{self.nodes[i]['model'].node_acronym} "
    #     return string_rep

    def function_wrapper(self, propagator, evaluator, exclude_locked=True):
        """ returns a function handle that accepts only parameters and returns the score. used to initialize the hessian analysis """

        def _function(_parameters, _graph, _propagator, _evaluator, _node_edge_index, _parameter_index):
            _graph.distribute_parameters_from_list(_parameters, _node_edge_index, _parameter_index)
            _graph.propagate(_propagator)
            score = _evaluator.evaluate_graph(_graph, _propagator)
            return score

        info = self.extract_attributes_to_list_experimental([], get_location_indices=True,
                                                                exclude_locked=exclude_locked)

        # def func(parameters):
        #     return _function(parameters, self, propagator, evaluator, info['node_edge_index'], info['parameter_index'])
        func = lambda parameters: _function(parameters, self, propagator, evaluator, info['node_edge_index'], info['parameter_index'])

        return func


    def initialize_func_grad_hess(self, propagator, evaluator, exclude_locked=True):
        self.func = self.function_wrapper(propagator, evaluator, exclude_locked=exclude_locked)
        self.grad = grad(self.func)
        hess_tmp = hessian(self.func) # hessian requires a numpy array, so wrap in this way
        # def hess_tmp(parameters):
        #     return hessian(self.func)(np.array(parameters))
        # self.hess = hess_tmp #lambda parameters: hess_tmp(np.array(parameters))
        self.hess = lambda parameters: hess_tmp(np.array(parameters))

        attributes = self.extract_attributes_to_list_experimental(['parameter_imprecisions'])
        parameter_imprecisions = np.expand_dims(np.array(attributes['parameter_imprecisions']), axis=1)
        scale_matrix = np.matmul(parameter_imprecisions, parameter_imprecisions.T)
        self.scaled_hess = lambda parameters: hess_tmp(np.array(parameters)) * scale_matrix
        return


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
    
    @property
    def propagate_on_edges(self):
        return self._propagate_on_edges

    def clear_propagation(self):
        self._propagator_saves = {}  # maybe this fixes weird, unphysical results from systems
        for node in self.nodes:
            if 'states' in self.nodes[node]:
                self.nodes[node].pop('states')

        for edge in self.edges:
            if 'states' in self.edges[edge]:
                self.edges[edge].pop('states')
        return
    
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
        
        AdditiveNoise.simulate_with_noise = start_noise_status # restore initial value
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

        AdditiveNoise.simulate_with_noise = start_noise_status # restore initial value
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
            if self._propagate_on_edges and 'model' in self.edges[edge]:  # this also simulates components stored on the edges, if there is a model on that edge
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
            ax[1].plot(propagator.f, 10 * np.log10(psd/noisy_max), ls=line, label=label)
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

        self._propagator_saves = {}  # will save each propagator here

        for node in self.propagation_order:  # loop through nodes in the prescribed, physical order
            if not self.pre(node):  # check if current node has any incoming edges, if not, pass the node the null input propagator directly
                tmp_states = [propagator.state]  # nodes take a list of propagators as default, to account for multipath
            else:  # if we have incoming nodes to get the propagator from
                tmp_states = []  # initialize list to add all incoming propagators to
                for edge in self.get_in_edges(node):  # loop through incoming edges
                    if self._propagate_on_edges and 'model' in self.edges[edge]:  # this also simulates components stored on the edges, if there is a model on that edge
                        signal = self.edges[edge]['model'].propagate(self._propagator_saves[edge], propagator, 1, 1, save_transforms=save_transforms)
                        
                        # add noise to propagation
                        noise_model = self.edges[edge]['model'].noise_model
                        if (noise_model is not None):
                            signal = noise_model.add_noise_to_propagation(signal, propagator)

                        tmp_states += signal
                    else:
                        tmp_states += self._propagator_saves[edge]

            # save the list of propagators at that node locations (deepcopy required throughout)
            states = self.nodes[node]['model'].propagate(tmp_states, propagator, self.in_degree(node), self.out_degree(node), save_transforms=save_transforms)

            # add noise to propagation
            noise_model = self.nodes[node]['model'].noise_model
            if (noise_model is not None):
                for i in range(len(states)):
                    states[i] = noise_model.add_noise_to_propagation(states[i], propagator)

            for i, (edge, state) in enumerate(zip(self.get_out_edges(node), states)):
                self._propagator_saves[edge] = [np.sqrt(self.coupling_efficiency) * state]  # we can use the edge as a hashable key because it is immutable (so we can use tuples, but not lists)

            self._propagator_saves[node] = states

        return self

    def measure_propagator(self, node):  # get the propagator state now that it is saved in a dictionary to avoid deepcopy
        return self._propagator_saves[node][0]

    def visualize_transforms_dof(self, ax, propagator, dof='f', label_verbose=1):
        for node in self.nodes():
            if self.nodes[node]['model'].transform is not None:
                for _, (dof_i, transform, label) in enumerate(self.nodes[node]['model'].transform):
                    if label_verbose == 0:
                        label_str = '{}'.format(label)
                    else:
                        label_str = 'Node {} | {} | {}'.format(node, self.nodes[node]['name'], label)

                    if (dof_i == dof) and (dof=='t'):
                        ax.plot(propagator.t, transform/np.max(transform), label=label_str)
                    elif (dof_i == dof) and (dof=='f'):
                        ax.plot(propagator.f, transform/np.max(transform), label=label_str)
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
                        ax[0].plot(propagator.t, transform/np.max(transform), label=label_str)
                    elif dof == 'f':
                        ax[1].plot(propagator.f, transform/np.max(transform), label=label_str)
        for axi in ax:
            axi.legend()
        ax[0].set_xlabel('Time')
        ax[1].set_xlabel('Frequency')

        scale_units(ax[0], unit='s', axes=['x'])
        scale_units(ax[1], unit='Hz', axes=['x'])

        plt.show()
        return

    def draw(self, ax=None, labels=None, legend=False, method='planar', ignore_warnings=True):
        """
        custom plotting function to more closely resemble schematic diagrams

        :param ax:
        :param labels:
        :return:
        """
        if ignore_warnings: warnings.simplefilter('ignore', category=(FutureWarning, cb.mplDeprecation))

        if ax is None:
            fig, ax = plt.subplots(1,1)

        if labels is None:
            labels = {node:f"{node}|{self.nodes[node]['model'].node_acronym}" for node in self.nodes}

        if method == 'planar':
            pos = nx.planar_layout(self)
        elif method == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self)
        else:
            pos = nx.planar_layout(self) # planar as default, though on occasion can fail with system graph

        nx.draw_networkx(self, ax=ax, pos=pos, labels=labels, alpha=1.0, node_color='darkgrey')

        if ignore_warnings: warnings.simplefilter('always', category=(FutureWarning, cb.mplDeprecation))
        return


    @property
    def propagation_order(self):
        """Returns the sorted order of nodes (based on which reverse walking the graph)
        """
        if nx.algorithms.recursive_simple_cycles(self):  # do not proceed if there is a cycle in the graph
            raise ValueError("There is a loop in the graph - current versions cannot simulate loops")

        _propagation_order = []  # initialize the list which defines the order nodes are visited
        node_set = set(self.nodes)  # use set type to compare which nodes have already been added
        for it in range(len(self.nodes)):  # maximum number of times the graph needs to be checked
            # for all nodes that haven't been added to _propagation_order, check which ones have no outgoing edges (terminal)
            terminal_nodes = [node for node in node_set if set(self.suc(node)).issubset(set(_propagation_order))]
            for terminal_node in terminal_nodes:
                _propagation_order = [terminal_node] + _propagation_order  # add these terminal nodes to the beginning of the order
            node_set = node_set - set(terminal_nodes)  # remove these terminal nodes from the our comparator set
        return _propagation_order

    @propagation_order.setter
    def propagation_order(self, propagation_order):
        self._propagation_order = propagation_order
        return

    def assert_number_of_edges(self):
        """Loops through all nodes and checks that the proper number of input/output edges are connected
        """
        # check for loops
        if nx.algorithms.recursive_simple_cycles(self):
            raise RuntimeError('There are loops in the topology')

        for node in self.nodes:
            number_input_edges, number_output_edges = len(self.pre(node)), len(self.suc(node))
            self.nodes[node]['model'].assert_number_of_edges(number_input_edges, number_output_edges)
        if self._propagate_on_edges:
            for edge in self.edges:
                if 'model' in self.edges[edge]:
                    self.edges[edge]['model'].assert_number_of_edges(1, 1)  # by definition, edges have 1 in, 1 out
        return
    
    def attempt_topology_fix(self):
        """
        Does not fix loops
        Goes through nodes with a nonphysical number of inputs/outputs and removes them. It keeps removing till we have hit a physical state again

        This is mostly meant to deal with the case where a node is stranded with no output (which can arise when removing interferometer paths)
        Results are not guaranteed, but it will never break a working graph
        """

        # 1. Check all nodes, if one is unphysical then
        # 2. Remove that node, and recurse through successors and predecessors (start with predecessors due to our target problem) removing each that ends up being unphysical
        # 3. Once that's done, check the whole graph again till it all passes
        while True:
            all_models_physical = True
            for node in self.nodes:
                try:
                    number_input_edges, number_output_edges = len(self.pre(node)), len(self.suc(node))
                    self.nodes[node]['model'].assert_number_of_edges(number_input_edges, number_output_edges)
                except TypeError:
                    all_models_physical = False
                    while (number_input_edges not in self.nodes[node]['model']._range_input_edges) or \
                          (number_output_edges not in self.nodes[node]['model']._range_output_edges):
                        next_node = self.pre(node)
                        current_node = node
                        self.remove_node(node)
                        current_node = next_node
                        if current_node == -1: # if we're trying to remove the top node
                            raise AssertionError('Topology could not be fixed')
                        number_input_edges, number_output_edges = len(self.pre(node)), len(self.suc(node))

            if all_models_physical:    
                break


    def sample_parameters(self, probability_dist='uniform', **kwargs):
        """ Samples new parameters for each node-type """
        attributes = ['parameters', 'lower_bounds', 'upper_bounds', 'data_types', 'step_sizes']
        model_attributes = self.extract_attributes_to_list_experimental(attributes, get_location_indices=True)
        parameters = self.sample_parameters_callable(model_attributes['parameters'], model_attributes['lower_bounds'],
                                                     model_attributes['upper_bounds'], model_attributes['data_types'],
                                                     model_attributes['step_sizes'], probability_dist=probability_dist, **kwargs)

        self.distribute_parameters_from_list(parameters,
                                             model_attributes['node_edge_index'],
                                             model_attributes['parameter_index'])
        return


    def sample_parameters_to_list(self, probability_dist='uniform', **kwargs):
        """ Samples new parameters for each node-type """

        # TODO Change this as well
        attributes = ['parameters', 'lower_bounds', 'upper_bounds', 'data_types', 'step_sizes']
        model_attributes = self.extract_attributes_to_list_experimental(attributes, get_location_indices=False)
        parameters = self.sample_parameters_callable(model_attributes['parameters'], model_attributes['lower_bounds'],
                                                     model_attributes['upper_bounds'], model_attributes['data_types'],
                                                     model_attributes['step_sizes'], probability_dist=probability_dist, **kwargs)

        return parameters
    
    def get_parameter_bounds(self):
        """
        Returns the lower bounds, upper bounds of parameters, in the order returned by all other functions

        :Return: <lower bounds (array)>, <upper bounds (array)>
        """
        attributes = ['lower_bounds', 'upper_bounds']
        model_attributes = self.extract_attributes_to_list_experimental(attributes, get_location_indices=False)
        
        return np.array(model_attributes['lower_bounds']), np.array(model_attributes['upper_bounds'])
    
    def get_parameter_info(self, exclude_locked=True):
        """
        Returns the node number and parameter name of each parameter, in the order returned by all other functions

        :Return: for each parameter: 'node A: type <ModelType>: param N <ParamName>'
        """
        info = self.extract_attributes_to_list_experimental(['parameter_names'], exclude_locked=exclude_locked)
        param_names = info['parameter_names']
        node_nums = info['node_edge_index']
        param_indices = info['parameter_index']
        
        param_infos = ''

        for i in range(len(param_indices)):
            node_num = node_nums[i]
            node_type = type(self.nodes[node_nums[i]]['model'])
            param_i = param_indices[i]
            param_name_i = param_names[i]
            info_string = f'node {node_num}, type {node_type}, param {param_i}={param_name_i}'
            param_infos += info_string + '\n'
        
        return param_infos

    @staticmethod
    def extract_attributes(_node_edge, _model, _attributes, _model_attributes, exclude_locked=True, *args):
        """ appends into _lst all unlocked attributes (from string _attribute) in _model
        args is always args[0] = node_edge_indices, and args[1] = parameter_indices
        """
        for i in range(_model.number_of_parameters):

            if _model.parameter_locks[i] and exclude_locked:
                continue

            for _attribute in _attributes:
                _model_attributes[_attribute].append(getattr(_model, _attribute)[i])

            if args:
                args[0].append(_node_edge)
                args[1].append(i)

        return


    def extract_attributes_to_list_experimental(self, attributes, get_location_indices=True, exclude_locked=True):
        """ experimental: to extract model variables to a list, based on a list of variables names """

        # first generate multiple empty lists for each attribute of attributes
        model_attributes = {}
        for attribute in attributes:
            model_attributes[attribute] = []

        if get_location_indices:
            model_attributes['node_edge_index'], model_attributes['parameter_index'] = [], []  # these will help translate from a list of parameters/parameter info to a graph structure


        # we loop through all nodes and add the relevant info (bounds, etc.)
        for node in self.nodes:
            model = self.nodes[node]['model']
            if model.node_lock and exclude_locked:
                continue
            if get_location_indices:
                self.extract_attributes(node, model, attributes, model_attributes, exclude_locked,
                                        model_attributes['node_edge_index'],
                                        model_attributes['parameter_index'])
            else:
                self.extract_attributes(node, model, attributes, model_attributes)


        # here, if considering edges too, we loop through and add them to each list (instead of a node hash, it is the edge tuple)
        if self._propagate_on_edges:
            for edge in self.edges:
                if 'model' in self.edges[edge]:
                    model = self.edges[edge]['model']
                    if model.node_lock and exclude_locked:
                        continue
                    if get_location_indices:
                        self.extract_attributes(edge, model, attributes, model_attributes, exclude_locked,
                                                model_attributes['node_edge_index'],
                                                model_attributes['parameter_index'])
                    else:
                        self.extract_attributes(edge, model, attributes, model_attributes)
        return model_attributes

    # TODO: we need to have this extract a dynamic selection (or always all) of the model characteristics (bounds, names, type, ...)
    def extract_parameters_to_list(self, exclude_locked=True):
        """ Extracts the current parameters, bounds and information for re-distributing from the graph structure """

        attributes = ['parameters', 'lower_bounds', 'upper_bounds']

        model_attributes = self.extract_attributes_to_list_experimental(attributes,
                                                                        get_location_indices=True,
                                                                        exclude_locked=exclude_locked)

        parameters = model_attributes['parameters']
        node_edge_index = model_attributes['node_edge_index']
        parameter_index = model_attributes['parameter_index']
        lower_bounds = model_attributes['lower_bounds']
        upper_bounds = model_attributes['upper_bounds']

        return parameters, node_edge_index, parameter_index, lower_bounds, upper_bounds

    def distribute_parameters_from_list(self, parameters, node_edge_index, parameter_index):
        """ from the lists created in 'extract_parameters_to_list', we distribute these (or altered versions, like in scipy.optimize.minimize) back to the graph"""
        for i, (parameter, node_edge, parameter_ind) in enumerate(zip(parameters, node_edge_index, parameter_index)):
            if type(node_edge) is tuple:  # then we know this is an edge
                self.edges[node_edge]['model'].parameters[parameter_ind] = parameter
            else:  # anything else is a node
                self.nodes[node_edge]['model'].parameters[parameter_ind] = parameter
        return

    def inspect_parameters(self):
        """ Loops through all nodes & edge (if enabled) and prints information about the parameters """
        for node in self.nodes:
            self.nodes[node]['model'].inspect_parameters()
        if self._propagate_on_edges:
            for edge in self.edges:
                if 'model' in self.edges[edge]:
                    self.edges[edge]['model'].inspect_parameters()

    def inspect_state(self, propagator, freq_log_scale=False, title=''):
        """ we loop through all nodes and plot the optical state *after* the node"""
        fig, ax = plt.subplots(2, 1)
        linestyles = cycle(['-', '--', '-.', ':'])

        # please note that we do not include the edges here (most of the time I doubt we will use edges, but it may be useful in the future)
        # for cnt, node in enumerate(self.propagation_order):
        for cnt, node in enumerate(reversed(self.propagation_order)):
            state = self.measure_propagator(node)
            line = {'ls':next(linestyles), 'lw':3}
            ax[0].plot(propagator.t, power_(state), label=self.nodes[node]['model'].__class__.__name__, **line)
            psd = psd_(state, propagator.dt, propagator.df)
            if freq_log_scale:
                ax[1].plot(propagator.f, 10 * np.log10(psd/np.max(psd)), **line)
            else:
                ax[1].plot(propagator.f, 0.1*cnt + psd/np.max(psd), **line)
        ax[0].legend()
        plt.title(title)
        plt.show()


    @staticmethod
    def sample_parameters_callable(parameters_current, lower_bounds, upper_bounds, data_types, step_sizes, probability_dist = 'uniform', **kwargs):
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
        model_edges = [(self.nodes[i]['model'].node_acronym, self.nodes[j]['model'].node_acronym) for (i, j, _) in self.edges]
        return nodes, edges, models, model_edges


#%% Sampling functions for mutation operations on parameters
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

    print('parameter_old {} | parameter {} | radius {} | left {} | right {}'.format(parameter_old, parameter, radius, left, right))

    return parameter
