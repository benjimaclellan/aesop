"""

"""

import autograd.numpy as np

import networkx as nx
import copy
import matplotlib.pyplot as plt
from itertools import cycle
import warnings

from lib.base_classes import Graph as GraphParent
from .assets.functions import power_, psd_


class Graph(GraphParent):
    """Parent class
    """

    __internal_var = None

    def __init__(self, nodes = dict(), edges = list(), propagate_on_edges = False, deep_copy=False):
        """
        """
        super().__init__()

        for node, model in nodes.items():
            self.add_node(node, **{'model': model, 'name': 'FEATURE NOT IMPLEMENTED', 'lock': False})

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

        self._deep_copy = deep_copy

        self._propagator_saves = {}

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

    def clear_propagation(self):
        for node in self.nodes:
            if 'states' in self.nodes[node]:
                self.nodes[node].pop('states')

        for edge in self.edges:
            if 'states' in self.edges[edge]:
                self.edges[edge].pop('states')
        return

    def propagate(self, propagator):
        if self._deep_copy:
            return self._propagate_deepcopy(propagator)
        
        # return self._propagate_limit_deepcopy(propagator)
        return self._propagate_no_deepcopy(propagator)

    def _propagate_no_deepcopy(self, propagator):
        """
        Uses no deepcopy functionality - hopefully avoiding issues with autograd.
        Instead, the propagator object at each nodes is saved to a separate dictionary, where the key is the node
        """

        print('This propagate function does NOT use copy.deepcopy')

        propagation_order = self.propagation_order
        self._propagator_saves = {}  # will save each propagator here ##TODO make sure this is not saving the same object in multiple places somehow (it doesn't seem like it though)

        for node in propagation_order:  # loop through nodes in the prescribed, physical order
            if not self.pre(node):  # check if current node has any incoming edges, if not, pass the node the null input propagator directly
                tmp_states = [propagator.state]  # nodes take a list of propagators as default, to account for multipath
            else:  # if we have incoming nodes to get the propagator from
                tmp_states = []  # initialize list to add all incoming propagators to
                for edge in self.get_in_edges(node):  # loop through incoming edges
                    if self._propagate_on_edges and 'model' in self.edges[edge]:  # this also simulates components stored on the edges, if there is a model on that edge
                        tmp_states += self.edges[edge]['model'].propagate(self._propagator_saves[edge], propagator, 1, 1)
                    else:
                        tmp_states += self._propagator_saves[edge]

            # save the list of propagators at that node locations (deepcopy required throughout)
            states = self.nodes[node]['model'].propagate(tmp_states, propagator, self.in_degree(node), self.out_degree(node))
            for i, (edge, state) in enumerate(zip(self.get_out_edges(node), states)):
                self._propagator_saves[edge] = [state]  # we can use the edge as a hashable key because it is immutable (so we can use tuples, but not lists)

            self._propagator_saves[node] = states
        return self

    def measure_propagator(self, node):  # get the propagator state now that it is saved in a dictionary to avoid deepcopy
        return self._propagator_saves[node][0]

    def _propagate_limit_deepcopy(self, propagator):
        """
        """
        propagation_order = self.propagation_order
        for node in propagation_order:
            # for all non-start node, the rep invariant is that it ALREADY contains a propagator (albeit not updated yet)s
            if not self.pre(node):
                tmp_states = [propagator.state] 
            else:
                tmp_states = []
                for edge in self.get_in_edges(node):
                    if self._propagate_on_edges and 'model' in self.edges[edge]:
                        # TODO: investigate this suspicious use of the previous node (should prob use )
                        tmp_states += self.edges[edge]['model'].propagate(self.nodes[edge[0]]['states'], propagator, 1, 1)  # TODO: Check that models on edge are single spatial mode maybe
                    else:
                        tmp_states += self.edges[edge]['states']
            
            states = self.nodes[node]['model'].propagate(tmp_states, propagator, self.in_degree(node), self.get_out_degree(node))
            for i, (edge, state) in enumerate(zip(self.get_out_edges(node), states)):
                if (i == 0):
                    self.edges[edge]['states'] = [state]
                else:
                    self.edges[edge]['states'] = copy.deepcopy([state])
            self.nodes[node]['states'] = states
            
        return self
    
    def _propagate_deepcopy(self, propagator):
        """
        """
        propagation_order = self.propagation_order
        for node in propagation_order:  # loop through nodes in the prescribed, physical order
            if not self.pre(node):  # check if current node has any incoming edges, if not, pass the node the null input propagator directly
                tmp_states = [copy.deepcopy(propagator.state)]  # nodes take a list of propagators as default, to account for multipath
            else:  # if we have incoming nodes to get the propagator from
                tmp_states = []  # initialize list to add all incoming propagators to
                for edge in self.get_in_edges(node):  # loop through incoming edges
                    if self._propagate_on_edges and 'model' in self.edges[edge]:  # this also simulates components stored on the edges, if there is a model on that edge
                        tmp_states += copy.deepcopy(self.edges[edge]['model'].propagate(self.nodes[edge[0]]['states'], propagator, 1, 1))  # TODO: Check that models on edge are single spatial mode maybe
                        # tmp_states += self.edges[edge]['model'].propagate(self.nodes[edge[0]]['states'], propagator, 1, 1)
                    else:
                        tmp_states += copy.deepcopy(self.edges[edge]['states'])
                        # tmp_states += self.edges[edge]['states']

            # save the list of propagators at that node locations (deepcopy required throughout)
            states = self.nodes[node]['model'].propagate(tmp_states, propagator, self.in_degree(node), self.out_degree(node))
            for i, (edge, state) in enumerate(zip(self.get_out_edges(node), states)):
                self.edges[edge]['states'] = copy.deepcopy([state])
                # self.edges[edge]['states'] = [state]

            self.nodes[node]['states'] = copy.deepcopy(states)
            # self.nodes[node]['states'] = states
        return self
            

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
        for node in self.nodes:
            number_input_edges, number_output_edges = len(self.pre(node)), len(self.suc(node))
            self.nodes[node]['model'].assert_number_of_edges(number_input_edges, number_output_edges)
        if self._propagate_on_edges:
            for edge in self.edges:
                if 'model' in self.edges[edge]:
                    self.edges[edge]['model'].assert_number_of_edges(1, 1)  # by definition, edges have 1 in, 1 out
        return

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

        model_attributes = self.extract_attributes_to_list_experimental(attributes, get_location_indices=True,
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

    def inspect_state(self, propagator):
        """ we loop through all nodes and plot the optical state *after* the node"""
        fig, ax = plt.subplots(2, 1)
        linestyles = cycle(['-', '--', '-.', ':'])

        # please note that we do not include the edges here (most of the time I doubt we will use edges, but it may be useful in the future)
        for node in reversed(self.propagation_order):
            state = self.measure_propagator(node)
            line = {'ls':next(linestyles), 'lw':3}
            ax[0].plot(propagator.t, power_(state), label=node, **line)
            ax[1].plot(propagator.f, psd_(state, propagator.dt, propagator.df), **line)

        ax[0].legend()
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
