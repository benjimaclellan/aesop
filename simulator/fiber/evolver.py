import autograd.numpy as np
from autograd import jacobian
import random
import matplotlib.pyplot as plt
import pickle
import copy
from autograd.numpy.numpy_boxes import ArrayBox
import itertools
import networkx as nx
import operator
import uuid

import simulator.fiber.assets.hessian_graph_analysis as hessian_analysis
from simulator.fiber.evolution_operators.evolution_operators import RemoveComponent
from .evolution_operators.evolution_operators import *
import config.config as configuration
from simulator.fiber.evolution_operators.evolution_operators import AddSeriesComponent, RemoveComponent, SwapComponent, AddParallelComponent

# TODO: fix the reinforcement learning one...
# TODO: stop regenerating the probability matrix each time: this is wasteful when we update it multiple times in a generation (i.e. for multiple children of the same graph)


class ProbabilityLookupEvolver(object):
    """
    Evolver object will take in and update graphs based on a stochastic probability matrix
    It will have a set of rules for updating a graph's stochastic matrix, in order to update the graph next time

    Matrices are defined as a probability matrix of Node/Edge versus Evolution Operator. p(node0, SwapNode) for example is the probability
    that the 'SwapNode' operator be applied on node node0. 

    The application of any operator must take the form apply_evolution_at(graph, node_or_edge, verbose=whatever)

    This class has the most basic probability selection: if a given operator can possibly be run on a node/edge, it will be assigned a value of one.
    If it is not possible, it is assigned a value of zero. Probability are normalized prior to selecting the operator
    """

    def __init__(self, verbose=False, debug=False, **attr):
        # self.generation = 0.0  # will be updated as the ratio between current_generation/total_generations
        self.verbose = verbose
        self.debug = debug
        self.evo_op_list = [evo_op(verbose=self.verbose) for evo_op in configuration.EVOLUTION_OPERATORS.values()] # generate all evolution operators
        super().__init__(**attr)
    
    def evolve_graph(self, graph, evaluator, generation=None):
        """
        Evolves graph according to the stochastic matrix probabilites

        :param graph: the graph to evolve
        :param evaluator: not used in the base implementation, but may be useful in the future
        :param generation: this is a ratio of the total generation (curr_gen/total_gens)
        """
        current_uuid, parent_uuid = copy.copy(graph.current_uuid), copy.copy(graph.parent_uuid)

        self.update_graph_matrix(graph, evaluator, generation=generation)

        if self.verbose:
            print(f'In the evolve_graph function : generation is {generation}')
            print(f'evolving graph:')
            print(graph)

        if self.debug:
            print(f'evolution probability matrix for graph {graph}')
            print(graph.evo_probabilities_matrix)

        node_or_edge, evo_op = graph.evo_probabilities_matrix.sample_matrix()
        graph = evo_op.apply_evolution(graph, node_or_edge)
        
        if self.verbose:
            print(f'\nevolving on: {node_or_edge}, with operator: {evo_op}\n')
        
        if self.verbose:
            print(f'evolved graph:')
            print(graph)
            print()
       
        graph.update_graph()

        x, *_, lower_bounds, upper_bounds = graph.extract_parameters_to_list()

        assert np.logical_and(lower_bounds <= x, x <= upper_bounds).all(), f'lower bound: {lower_bounds}\n params: {x}\n upperbounds: {upper_bounds}' #' \n pre-swap param: {pre_swap_params}\n new_node params: {list(zip(new_model.parameter_names, new_model.parameters))}'
        print(f'Evolver has chosen: {evo_op} | {node_or_edge}')
        # print(graph.nodes[node_or_edge.node])

        graph.parent_uuid = current_uuid
        graph.current_uuid = uuid.uuid4()
        graph.latest_mutation = evo_op.__class__.__name__

        return graph, evo_op

    def create_graph_matrix(self, graph, evaluator):
        """
        :param graph: the graph to evolve
        :param evaluator: not used in the base implementation, but may be useful in the future

        TODO: can we refactor to only add possible locations and evo ops? But without traversing more...
        """
        node_pair_list = list(itertools.combinations(nx.algorithms.dag.topological_sort(graph), 2))
        graph.evo_probabilities_matrix = self.ProbabilityMatrix(self.evo_op_list, list(graph.nodes), list(graph.edges), graph.interfaces, node_pair_list)
        for evo_op in self.evo_op_list:
            locations = evo_op.possible_evo_locations(graph)
            for loc in locations:
                graph.evo_probabilities_matrix.set_prob_by_nodeEdge_op(1, loc, evo_op)
        
        graph.evo_probabilities_matrix.normalize_matrix()
        graph.evo_probabilities_matrix.verify_matrix()

    def update_graph_matrix(self, graph, evaluator, generation=None):
        """
        Function does not use evaluator in this implementation. However, they may be valid parameters to modify later
        """
        self.create_graph_matrix(graph, evaluator) # just fully remakes the graph matrix for now

    def random_graph(self, graph, evaluator, propagator=None, n_evolutions=10, view_evo=False):
        for n in range(n_evolutions):
            if self.verbose:
                print(f'Starting evolution number {n}')
            try:
                graph_tmp, evo_op = self.evolve_graph(graph, evaluator, generation=n)
                # graph_tmp.assert_number_of_edges()
                graph = graph_tmp
                if view_evo:
                    graph.draw()
                    plt.show()
            except AssertionError as e:
                print(e)

        return graph
    
    def close(self):
        pass

    @staticmethod
    def possible_operators():
        return [evo_op for evo_op in configuration.EVOLUTION_OPERATORS.values()]

    class ProbabilityMatrix(object):
        """
        Probability matrix of (node/edge) v. Operator. The probability p(node_or_edge, operator) is the odd of operator being applied on node_or_edge

        Note that updates to the matrix can be done either through the provided function, or by direct manipulation of self.matrix
        """
        def __init__(self, op_list, node_list, edge_list, node_pairs_list, interface_list):
            self.op_to_index = {op: i for (i, op) in enumerate(op_list)}
            self.index_to_op = {i: op for (i, op) in enumerate(op_list)}
            self.node_or_edge_to_index = {node_or_edge: i for (i, node_or_edge) in enumerate(node_list + edge_list + node_pairs_list + interface_list)}
            self.index_to_node_or_edge = {i: node_or_edge for (i, node_or_edge) in enumerate(node_list + edge_list + node_pairs_list + interface_list)}

            self.matrix = np.zeros((len(self.node_or_edge_to_index), len(self.op_to_index)))
    
        def normalize_matrix(self):
            self.matrix = self.matrix / np.sum(self.matrix)
        
        def verify_matrix(self):
            if not np.isclose(np.sum(self.matrix), 1):  # matrix sums to proper probability
                # raise ValueError(f'The matrix does not sum to 1. Instead sums to {np.sum(self.matrix)}\n{self.matrix}')
                raise RuntimeWarning(f'The matrix does not sum to 1. Instead sums to {np.sum(self.matrix)}\n{self.matrix}')

            if not ((self.matrix <= 1).all() and (self.matrix >= 0).all()):  # assert all probabilities are between 0 and 1
                # raise ValueError(f'Matrix elements not proper probabilities. {self.matrix}')
                raise RuntimeWarning(f'Matrix elements not proper probabilities. {self.matrix}')

        def sample_matrix(self):
            flat_matrix = self.matrix.flatten()
            flat_indices = np.arange(0, flat_matrix.shape[0], step=1)
            flat_index = np.random.choice(flat_indices, p=flat_matrix)

            # convert flat index to proper bidirectional index
            row = flat_index // self.matrix.shape[1]
            col = flat_index % self.matrix.shape[1]

            return self.index_to_node_or_edge[row], self.index_to_op[col]
        
        def set_prob_by_nodeEdge_op(self, probability, node_or_edge, op):
            self.matrix[self.node_or_edge_to_index[node_or_edge], self.op_to_index[op]] = probability
        
        def get_prob_by_nodeEdge_op(self, node_or_edge, op):
            return self.matrix[self.node_or_edge_to_index[node_or_edge], self.op_to_index[op]]
        
        def get_probs_at_node_or_edge(self, node_or_edge):
            """
            Warning: modifying the returned vector will modify matrix values
            """
            return self.matrix[self.node_or_edge_to_index[node_or_edge], :]
        
        def get_probs_at_operator(self, op):
            """
            Warning: modifying the returned vector will modify matrix values
            """
            return self.matrix[:, self.op_to_index[op]]
        
        def set_probs_at_operator(self, op, probs):
            self.matrix[:, self.op_to_index[op]] = probs
        
        def get_largest_prob_node_or_edge_and_op(self):
            """
            Returns the node/edge and operator associated with the largest value. If multiple node/edge and operator
            pairs have the same value, returns a random pair (all same-valued options are returned with equal probability)

            :Pre-condition: the matrix is populated
            """
            max = np.amax(self.matrix)
            locations = np.where(np.isclose(self.matrix, max))
            possibilities = list(zip(locations[0].tolist(), locations[1].tolist()))
            (row, col) = random.choice(possibilities)
            return self.index_to_node_or_edge[row], self.index_to_op[col]

        def __str__(self):
            string_rep = f'operator to index: {self.op_to_index}\n' + f'node/edge to index: {self.node_or_edge_to_index}\n' + \
                         f'matrix: \n{self.matrix}'
            return string_rep


class OperatorBasedProbEvolver(ProbabilityLookupEvolver):
    """
    This evolver allows us to set the relative probabilities of each operator (e.g. we could set p = 0.25 that each operator is selected)
    In the case that an operator cannot be applied, the remaining probability is distributed to all other operator probabilities such that
    their ratios remain the same.

    Note that in this implementation, each possible location for a given evolution operator is selected with equal probability
    """
    def __init__(self, op_to_prob=None, **attr):
        """
        Sets up the operator based probability lookup evolver
        TODO: add an option to make the operator based probability time variant

        :param op_to_prob: a dictionary mapping an operator class to a probability. Note that the sum of probabilities DOES NOT need to be one. Normalization is handled in code
        """
        super().__init__(**attr)
        if op_to_prob is None:
            number_evo_ops = len(configuration.EVOLUTION_OPERATORS)
            self.op_to_prob = {evo_op: 1 / number_evo_ops for evo_op in configuration.EVOLUTION_OPERATORS.values()}
        else:
            self.op_to_prob = op_to_prob
    
    def create_graph_matrix(self, graph, evaluator):
        super().create_graph_matrix(graph, evaluator)
        for evo_op in self.evo_op_list:
            op_probs = graph.evo_probabilities_matrix.get_probs_at_operator(evo_op)
            op_probs = self.scale_within_operator(graph, evo_op, op_probs)
            sum_probs = np.sum(op_probs)
            if np.sum(op_probs) != 0:
                graph.evo_probabilities_matrix.set_probs_at_operator(evo_op, op_probs / sum_probs * self.op_to_prob[evo_op.__class__])
        
        graph.evo_probabilities_matrix.normalize_matrix()
        graph.evo_probabilities_matrix.verify_matrix()
    
    def scale_within_operator(self, graph, evo_op, op_probs):
        """
        Allows the relative probabilities of locations (for a single evolution operators) to be adjusted

        Default implementation leaves each location with equal probability
        """
        return op_probs


class HessianProbabilityEvolver(OperatorBasedProbEvolver):
    # TODO: add a constructor where the exact formula by which relative removal location probabilities are scaled can be set?
    def __init__(self, probability_flattening=3, max_prob_ratio=10, **attr):
        """
        TODO: make probability_flattening also (potentially) generation dependent
        :param probability_flattening: the larger probability_flattening, the less the "free-wheeling" differences affects the probabilities
                                       if the base relative probabilities coming out of the hessian are [a, b, c], the probabilities used are
                                       [a**(1/probability_flattening), b**(1/probability_flattening), c**(1/probability_flattening)]
        :param max_prob_ratio: maximum ratio of probabilities allowed within a single operator
        """
        op_to_prob = {}
        probs = {'AddSeriesComponent': 1, 'AddParallelComponent': 1, 'RemoveComponent': 4, 'SwapComponent': 1}
        for evo_op_id, evo_op in configuration.EVOLUTION_OPERATORS.items():
            op_to_prob[evo_op] = probs[evo_op_id]

        super().__init__(op_to_prob=op_to_prob, **attr)
        self.probability_flattening = probability_flattening
        self.max_prob_ratio = max_prob_ratio

    def update_graph_matrix(self, graph, evaluator, generation=None):
        """
        Function does not use evaluator in this implementation. However, they may be valid parameters to modify later
        """
        if generation is not None:
            # here we will update the generation-dependent probabilities for each operator
            probs = {'AddSeriesComponent': 1-generation,
                     'AddParallelComponent': 1-generation,
                     'RemoveComponent': 2*generation,
                     'SwapComponent': 1}
            for evo_op_id, evo_op in configuration.EVOLUTION_OPERATORS.items():
                self.op_to_prob[evo_op] = probs[evo_op_id]

        if self.debug: print(self.op_to_prob)

        self.create_graph_matrix(graph, evaluator)  # just fully remakes the graph matrix for now

    def scale_within_operator(self, graph, evo_op, op_probs):
        if type(evo_op) == RemoveComponent:  # is hardcoded to our evo operators, can't REALLY be helped

            # method = 'proportionate'
            method = 'tournament'

            if method == 'proportionate':
                op_probs = self.hessian_proportionate(graph, op_probs)

            elif method == 'tournament':
                op_probs = self.hessian_tournament(graph, op_probs)

        return op_probs

    def hessian_tournament(self, graph, op_probs):
        edge_free_wheeling_scores = hessian_analysis.get_all_edge_scores(graph, as_log=False)

        # first we only go through and find the valid operations, and put them in a list to do tournament selection on
        possible_selections = {}
        for i in range(len(op_probs)):
            if np.isclose(op_probs[i], 0):  # probability should stay zero, if operation is not possible
                continue
            try:
                location = graph.evo_probabilities_matrix.index_to_node_or_edge[i]  # locations should be interfaces
                possible_selections[i] = edge_free_wheeling_scores[location.edge]
                # print(edge_free_wheeling_scores[location.edge])
                # print(f'\n\ncurrent location: {location}')
                # print(f'current edge: {location.edge}')
                # print(f'flattened_prob: {op_probs[i]}\n\n')
            except KeyError:
                pass  # is fine, lots of locations aren't edges

        # if we have at least one valid place to remove, we do tournament selection on them
        if len(possible_selections) > 0:
            k_percentage = 0.5  # as the number of locations is changing, we make it percentage at first
            k = int(round(k_percentage * len(possible_selections.values())))  # size of tournament
            if k < 1: # quick checks to make sure k is okay
                k = 1
            if k > len(possible_selections.values()):
                k = len(possible_selections.values())
            full_tour = list(possible_selections.keys())  # all possible entries into the tournament
            random.shuffle(full_tour)  # shuffle, and take first
            tournament = {key: value for (key, value) in possible_selections.items() if key in full_tour[:k]}
            least_sensitive_in_tournament = min(tournament.items(), key=operator.itemgetter(1))[0]

            op_probs = [0 for i in range(len(op_probs))]
            op_probs[least_sensitive_in_tournament] = 1
            if self.debug:
                print('possible selections', possible_selections)
                print('k', k)
                print(tournament)
                print('least_sensitive', least_sensitive_in_tournament)
                print(op_probs)

        return op_probs

    def hessian_proportionate(self, graph, op_probs):
        edge_free_wheeling_scores = hessian_analysis.get_all_edge_scores(graph, as_log=False)

        score_array = np.array(list(edge_free_wheeling_scores.values()))
        reference = np.sum(score_array)

        for i in range(len(op_probs)):
            if np.isclose(op_probs[i], 0):  # probability should stay zero, if operation is not possible
                continue
            try:
                location = graph.evo_probabilities_matrix.index_to_node_or_edge[i]  # locations should be interfaces
                op_probs[i] = np.log10(reference / edge_free_wheeling_scores[location.edge]) ** (
                            1 / self.probability_flattening)
                # print(f'\n\ncurrent location: {location}')
                # print(f'current edge: {location.edge}')
                # print(f'flattened_prob: {op_probs[i]}\n\n')
            except KeyError:
                pass  # is fine, lots of locations aren't edges

        # limit max difference in probability
        if np.count_nonzero(op_probs) != 0:
            max_allowed_prob = np.min(op_probs[np.nonzero(op_probs)]) * self.max_prob_ratio
            for i in range(len(op_probs)):
                if op_probs[i] > max_allowed_prob:
                    op_probs[i] = max_allowed_prob
        return op_probs


    def random_graph(self, graph, evaluator, propagator=None, n_evolutions=10, view_evo=False):
        for n in range(n_evolutions):
            if self.verbose:
                print(f'Starting evolution number {n}')
            try:
                graph.initialize_func_grad_hess(propagator, evaluator)
                graph_tmp, evo_op = self.evolve_graph(graph, evaluator, generation=n)
                # graph_tmp.assert_number_of_edges()
                graph = graph_tmp
                if view_evo:
                    graph.draw()
                    plt.show()
            except AssertionError as e:
                print(e)

        return graph





# class SizeAwareLookupEvolver(ProbabilityLookupEvolver):
#     """
#     The size aware stochastic matrix evolver updates probabilities of certain operators being used based on the size of the graph.
#     The algorithm considers "growth" operators (add nodes or edges), and "reduction" operators (remove nodes or edges). Any operators not marked
#     as such via decorators will be considered neutral operators, which do not affect the graph complexity (e.g. SwapNode)
#
#     The probability rules are defined as such:
#     1. If the 'verify_operator_at' methods return false, the probability of an operator at a given node/edge is zero regardless of other rules
#     2. Each object has a 'ideal_node_num' parameter. This is the node number at which growth, reduction, and neutral operators have equal probability of being selected
#             - This number can be updated whenever
#     3. Let's define likelihood as probability before normalization. This evolver assigns a likelihood of 1 for neutral operators, 1 - alpha to growth operators,
#        and 1 + alpha to reduction operators. Alpha is clipped at +=0.9 by default, but can be clipped at any amplitude between 0 and 1 if desired
#     4. alpha = alpha(delta) is an odd function (where alpha < 0 where delta < 0), where delta = <# of nodes in the graph> - <ideal # of nodes>. alpha(0) = 0
#     """
#     def __init__(self,ideal_edge_num=10, alpha_func=lambda delta: (delta / 10)**3, alpha_bound=0.9, **attr):
#         """
#         Initializes SizeAwareMatrixEvolver (see class description above)
#
#         :param verbose: if True, evolution operators type and node/edges are printed to stdout. Otherwise, the evolution is 'silent'
#         :param ideal_node_number: number of nodes in a graph at which the growth and reduction evolution operators have equal chances of being selected
#         :param alpha_func: requirements are alpha(delta) is odd, alpha(0) = 0, alpha(delta < 0) < 0. Current function is selected such that alpha(10) = 1 (max bias)
#                            and such that the bias in likelihood be small if delta is small.
#         :param alpha_bound: amplitude at which to clip . Also supports alpha_bound as a tuple (e.g (-0.8, 0.9)). Note that alpha_bound=A is identical to alpha_bound=(-A, A)
#
#         Only the ideal node number is intended to be updated over the course of the object's existence (there's a useful case for it I can see, unlike the other params)
#         That said, updates to any of these parameters is possible (i.e. the code will use new alpha_func and alpha_max if their values are changed)
#         """
#         super().__init__(**attr)
#         self.ideal_edge_num = ideal_edge_num
#         self.alpha_func = alpha_func
#
#         # verify that the alpha_max input is valid. We'd verify for alpha_funct too but it's too complicated so
#         if type(alpha_bound) is int or type(alpha_bound) is float:
#             if alpha_bound < 0 or alpha_bound > 1:
#                 raise ValueError(f'alpha_bound given as a float/int must be in [0, 1]. {alpha_bound} is not valid')
#             self.alpha_bound = (-alpha_bound, alpha_bound)
#         else:
#             if alpha_bound[0] > alpha_bound[1]:
#                 raise ValueError(f'alpha_bound upper bound {alpha_bound[1]} must be greater than alpha_bound lower bound {alpha_bound[0]}')
#             elif -1 < alpha_bound[0] or alpha_bound[0] > 0:
#                 raise ValueError(f'Lower bound value for alpha_bound {alpha_bound[0]} must be in [-1, 0]')
#             elif 0 < alpha_bound[1] or alpha_bound[1] > 1:
#                 raise ValueError(f'Upper bound value for alpha_bound {alpha_bound[1]} must be in [0, 1]')
#             self.alpha_bound = alpha_bound
#
#     def create_graph_matrix(self, graph, evaluator):
#         """
#         :param graph: the graph to evolve
#         :param evaluator: not used in this implementation, but may be useful in the future
#         """
#         super().create_graph_matrix(graph, evaluator)
#         alpha = self._get_alpha_offset(graph)
#         for location in graph.evo_probabilities_matrix.node_or_edge_to_index.keys():
#             for op in graph.evo_probabilities_matrix.op_to_index.keys():
#                 evo_possible = graph.evo_probabilities_matrix.get_prob_by_nodeEdge_op(location, op) != 0
#                 if evo_possible:
#                     if op.__class__.__name__ in configuration.GROWTH_EVO_OPERATORS.keys():
#                         likelihood = 1 - alpha
#                     elif op.__class__.__name__ in configuration.REDUCTION_EVO_OPERATORS.keys():
#                         likelihood = 1 + alpha
#                     else:
#                         likelihood = 1
#                     graph.evo_probabilities_matrix.set_prob_by_nodeEdge_op(likelihood, location, op)
#
#         graph.evo_probabilities_matrix.normalize_matrix()
#         graph.evo_probabilities_matrix.verify_matrix()
#
#     def _get_alpha_offset(self, graph):
#         delta = len(graph.edges) - self.ideal_edge_num
#         return np.clip(self.alpha_func(delta), self.alpha_bound[0], self.alpha_bound[1])
#





# class ReinforcementLookupEvolver(ProbabilityLookupEvolver):
#     """
#     Loosely based on reinforcement learning ideas, where only the operator selection is reinforcement based (node selection remains arbitrary)
#     The "state" of the graph is solely defined by the number of nodes, for simplicity

#     1. The probability matrix is arbitrary if we have no past history for a graph of size N (number of nodes)
#     2. Otherwise, the best historical solution op is assigned probability 1 - epsilon (equally divided between all possible node/edges),
#        with equal probability across all other possible node/op pairs (epsilon can be gen dependent)
    
#     Implementation notes:
#     1. Each graph can store its previous scores / averages? At the beginning of each new optimization we can compute the updated "value"
#     2. Consider: how do we backpropagate value between states. For now we won't, we just calculate improvement in score from last time

#     The philosophy for now is: immediate improvement is the name of the game. We define the value of an operation as its predicted ,
#     where reward is improvement in score. The predicted improvement is aggregated from all graphs that the evolver is called on

#     # TODO: consider making value less based on immediate reward
#     # TODO: consider decaying epsilon scheme instead of constant epsilon value
#     """
#     def __init__(self, starting_value_matrix=None, epsilon=0.4, **attr):
#         """
#         Creates a ReinforcementMatrixEvolver (reinforcement is used a bit loosely here)

#         :param verbose: if True, evolution operators type and node/edges are printed to stdout. Otherwise, the evolution is 'silent'
#         :param starting_value_matrix: initial matrix of expected awards for different operators
#                                       Parameter is a tuple: (dictionary of numpy 2D vectors, ordered list of operators)
#                                       Dictionary key: number of nodes in graph, N
#                                       Dictionary value: row1 = numpy array of expected rewards (i.e. values) per operator (in same order as the ordered list)
#                                                         row2 = numpy array of integers, with number of times the same-index value of expected rewards has been updated in the past (required to tally long-term average)
#                                       If not provided, all configuration operators are selected, and all values are set to 0 to start

#                                       If a string, this is the string to a pickle file of a previous run
#         """
#         super().__init__(**attr)
#         self.epsilon = epsilon
        
#         if starting_value_matrix is None:
#             self.value_matrix = {} # value matrix will be consistently updated
#         elif type(starting_value_matrix) == str:
#             with open(starting_value_matrix, 'rb') as handle:
#                 self.value_matrix = pickle.load(handle)
#         else:
#             self.value_matrix = starting_value_matrix[0]
#             self.evo_op_list = starting_value_matrix[1] # override default in constructor
    
#     def create_graph_matrix(self, graph, evaluator):
#         self._update_value_matrix(graph)
#         self._translate_values_to_probabilities(graph, evaluator)

#     def _update_value_matrix(self, graph):
#         if len(graph.nodes) not in self.value_matrix: # check whether we already have an entry for this graph state
#             self.value_matrix[len(graph.nodes)] = np.zeros((2, len(self.evo_op_list)))

#         if graph.last_evo_record is not None and graph.score is not None and graph.last_evo_record['score'] is not None: # we can only update our evo matrices if we have a previous score to compare to
#             N = graph.last_evo_record['state']
#             reward = graph.last_evo_record['score'] - graph.score
#             evo_index = self.evo_op_list.index(graph.last_evo_record['op'])

#             # new value is the new average reward. If the previous average is Qk, and the new reward is Rk:
#             # average reward Q(k + 1) = Qk + (1/k)(R_k - Q_k)
#             self.value_matrix[N][1][evo_index] += 1
#             Qk = self.value_matrix[N][0][evo_index]
#             k = self.value_matrix[N][1][evo_index]
#             new_value = Qk + (1 / k) * (reward - Qk)

#             self.value_matrix[N][0][evo_index] = new_value

    
#     def _translate_values_to_probabilities(self, graph, evaluator):
#         """
#         Turns the value matrix into a matrix of probabilities with the following rules:
#         1. Total probability of the greedy operator being selected is 1 - epsilon
#         2. Probability of greedy operator is evenly dispersed among each node/edge where it can be applied
#         3. Remaining probability is distributed equally among all other possible operator-node/edge combinations

#         Pre-condition: assumes self.value_matrix is fully updated
#         """
#         super().create_graph_matrix(graph, evaluator) # get basic matrix with 1 for possible combos and 0 for impossible combos

#         greedy_op, greedy_op_num_possible = self._get_greedy_op_and_possible_num(graph)
#         greedy_op_prob = (1 - self.epsilon) / greedy_op_num_possible
#         other_op_prob = self.epsilon / (np.count_nonzero(graph.evo_probabilities_matrix.matrix) - greedy_op_num_possible)
        
#         for node_or_edge in list(graph.nodes) + list(graph.edges):
#             for op in self.evo_op_list:
#                 if not np.isclose(0, graph.evo_probabilities_matrix.get_prob_by_nodeEdge_op(node_or_edge, op)): # i.e. if an operation is possible
#                     prob = greedy_op_prob if op == greedy_op else other_op_prob
#                     graph.evo_probabilities_matrix.set_prob_by_nodeEdge_op(prob, node_or_edge, op)

#         # omit normalization because evo matrix should be normalized already. If not, we want it to crash, not silently fail
#         graph.evo_probabilities_matrix.verify_matrix()

#     def _get_greedy_op_and_possible_num(self, graph):
#         """
#         Returns the greedy operator. If there exist multiple potential greedy operators, it randomly selects one
#         """
#         # TODO: refactor this, it's ugly code...
#         # 1. Look through all max valued options, going in descending order of value (in case the max valued options are not applicable)
#         usable = np.ones_like(self.value_matrix[len(graph.nodes)][0])
#         while np.count_nonzero(usable) > 0:
#             max_val = np.max(np.array([self.value_matrix[len(graph.nodes)][0][i] for i in range(len(usable)) if usable[i]]))
#             indices = [i for (i, val) in enumerate(self.value_matrix[len(graph.nodes)][0]) if np.isclose(val, max_val)]
#             while len(indices) > 0:
#                 index = np.random.choice(indices)
#                 op = self.evo_op_list[index]
#                 greedy_op_num_possible = np.count_nonzero(graph.evo_probabilities_matrix.get_probs_at_operator(op))
#                 if greedy_op_num_possible > 0:
#                     return op, greedy_op_num_possible
#                 else:
#                     indices.remove(index)
            
#             for i in range(len(usable)):
#                 if np.isclose(max_val, self.value_matrix[len(graph.nodes)][0][i]):
#                     usable[i] = 0

#         # 2. no max valued option is available
#         raise ValueError('No possible operator found')

#     def evolve_graph(self, graph, evaluator, generation=None):
#         """
#         Evolves graph according to the stochastic matrix probabilites
#         The probabilities for operators are based on a epsilon-greedy learning method
#         The probabilities for different nodes with the same operator are equal, assuming the operation is possible

#         :param graph: the graph to evolve
#         :param evaluator: not used in the base implementation, but may be useful in the future
#         """
#         pre_evo_state = len(graph.nodes)
#         graph, evo_op = super().evolve_graph(graph, evaluator, generation=generation)
#         if graph.last_evo_record is None:
#             graph.last_evo_record = {}
#         graph.last_evo_record['op'] = evo_op
#         graph.last_evo_record['score'] = graph.score
#         graph.last_evo_record['state'] = pre_evo_state

#         return graph, evo_op
    
#     def close(self):
#         with open(f'reinforcement_evolver_value_matrix.pkl', 'wb') as handle:
#             pickle.dump(self.value_matrix, handle)


# class EGreedyHessianEvolver(ProbabilityLookupEvolver):
#     """
#     The epsilon-greedy Hessian evolver assigns value to different actions (action = an operator and the node/edge on which it applied)
#     based on the 0th order and 2nd order derivatve information of our fitness function with respect to the parameter space. 

#     Once all actions are assigned a value, it executes the action with the largest value with probability 1 - epsilon, and a random
#     action with probability epsilon. EPSILON CAN (and likely should) BE A GENERATION DEPENDENT FUNCTION.

#     The value of an action is determined by a relevent base score i.e. log10(free wheeling node score) = W, log10(terminal node score) = T, multiplied by
#     a scaling coefficient (which weighs importance of various factors). These scaling coefficients are hyperparameters, to be tuned as needed.

#     The action-node/edge values, V(O, N) are assigned as such, with floats a, b, c, d, e >= 0 as the scaling coefficients
    
#     Remove Path (on source/sink node, i.e. multipath): V = -a * T, a = <terminal_path_removal_coeff>
#         - If a multipath node is terminal, it's likely directing all the signal to one branch or another.
#           Therefore it should be likely we remove the other branch
#     Remove Path (on single-path node): V = max(-a * T_source * W, -a * T_sink * W)
#         - If we have a long path with a terminal source, we axe it. How do we know we're on the right branch?
#           Well the nodes should be free-wheeling if the source is terminal pointing away from this branch!
#     Remove Node (on single-path node): V = max(-c * W, -d * T>, c = <freewheel_removal_coeff>, d = terminal_removal_coeff
#         - A free-wheeling node is likely doing nothing, so might as well boot it
#         - Also, a terminal node Might be in passive state, so we can try booting it as well

#     TODO: implement later if it still seems like a good idea
#     Duplicate Node (on single-path node): V = -b * T, b = <terminal_duplicate_coeff>
#         - If a node is terminal, it's possible that it just needs more JUICE rather than being quasi-passive
#           (e.g. an EDFA might be at max gain, so chaining them might help!).
#           So we can also try duplicating that component to beef things up

#     Other operators: V = e = <default_val>
#     """
#     def __init__(self, epsilon=1, freewheel_removal_coeff=1, terminal_removal_coeff=0.5, terminal_duplicate_coeff=1,
#                 freewheel_path_removal_coeff=0.3, terminal_path_removal_coeff=5, default_val=1, **attr):
#         """
#         Creates a Hessian based evolver (which simplifies graphs based on the Hessian)

#         :param epsilon: probability epsilon with which a random node/edge, operator combo is selected
#                         epsilon can be a number or a function of the generation
#         :param freewheel_removal_coeff: hyperparameter which scales the base value for removing a free-wheeling node

#         """
#         if type(epsilon) == float or type(epsilon) == int:
#             self.epsilon = epsilon
#         else:
#             AssertionError('function based epsilon not implemented yet')
#             self.epsilon = epsilon
        
#         # hyperparameters
#         self.freewheel_removal_coeff = freewheel_path_removal_coeff
#         self.terminal_removal_coeff = terminal_removal_coeff
#         self.terminal_duplicate_coeff = terminal_duplicate_coeff
#         self.freewheel_path_removal_coeff = freewheel_path_removal_coeff
#         self.terminal_path_removal_coeff = terminal_path_removal_coeff
#         self.default_val = default_val

#         super().__init__(**attr)
    

#     def create_graph_matrix(self, graph, evaluator):
#         """
#         :param graph: the graph to evolve
#         :param evaluator: not used in this implementation, but may be useful in the future
#         """
#         super().create_graph_matrix(graph, evaluator) # get basic matrix with 1 for possible combos and 0 for impossible combos
        
#         x, *_ = graph.extract_parameters_to_list()

#         value_matrix = self._get_action_value_matrix(graph, evaluator) # we don't actually need the matrix, just the top val.
#                                                                        # BUT it's useful to see the full matrix for tuning/debugging purposes 
#         greedy_node_or_edge, greedy_op = value_matrix.get_largest_prob_node_or_edge_and_op()

#         non_greedy_prob = self.epsilon / np.sum(graph.evo_probabilities_matrix.matrix)

#         for node_or_edge in list(graph.nodes) + list(graph.edges):
#             for op in self.evo_op_list:
#                 if graph.evo_probabilities_matrix.get_prob_by_nodeEdge_op(node_or_edge, op) != 0:
#                     if node_or_edge == greedy_node_or_edge and op == greedy_op:
#                         likelihood = 1 - self.epsilon + non_greedy_prob # we can save the generation on the graph I guess
#                     else:
#                         likelihood = non_greedy_prob
#                 else: 
#                     likelihood = 0
                
#                 graph.evo_probabilities_matrix.set_prob_by_nodeEdge_op(likelihood, node_or_edge, op)
        
#         graph.evo_probabilities_matrix.normalize_matrix()
#         graph.evo_probabilities_matrix.verify_matrix()
#         if self.debug:
#             print(f'value matrix:')
#             print(value_matrix)

#     def _get_action_value_matrix(self, graph, evaluator):
#         """
#         The matrix is actually not needed for the implementation, but it's def needed for tuning hyperparameters and debugging
        
#         The action-node/edge values, V(O, N) are assigned as such, with floats a, b, c, d, e >= 0 as the scaling coefficients
        
#         Remove Path (on source/sink node, i.e. multipath): V = -a * T, a = <terminal_path_removal_coeff>
#             - If a multipath node is terminal, it's likely directing all the signal to one branch or another.
#             Therefore it should be likely we remove the other branch
#         Remove Path (on single-path node): V = max(-a * T_source * W, -a * T_sink * W)
#             - If we have a long path with a terminal source, we axe it. How do we know we're on the right branch?
#             Well the nodes should be free-wheeling if the source is terminal pointing away from this branch!
#         Remove Node (on single-path node): V = max(-c * W, -d * T>, c = <freewheel_removal_coeff>, d = terminal_removal_coeff
#             - A free-wheeling node is likely doing nothing, so might as well boot it
#             - Also, a terminal node Might be in passive state, so we can try booting it as well
#         """
#         # ------------ test to see if it'll crash on the first grad and hess
#         x, *_ = graph.extract_parameters_to_list()
#         graph.grad(x)
#         graph.hess(x)
#         # --------------- end test

#         terminal_node_scores, free_wheeling_node_scores = hessian_analysis.get_all_node_scores(graph)

#         value_matrix = self.ProbabilityMatrix(self.evo_op_list, list(graph.nodes), list(graph.edges)) # using this to build our value matrix

#         for node_or_edge in list(graph.nodes) + list(graph.edges):
#             for op in self.evo_op_list:
#                 if graph.evo_probabilities_matrix.get_prob_by_nodeEdge_op(node_or_edge, op) != 0:
#                     if op in configuration.PATH_REDUCTION_EVO_OPERATORS.values():
#                         if EGreedyHessianEvolver._is_multipath_node(graph, node_or_edge):
#                             val = -1 * self.terminal_path_removal_coeff * terminal_node_scores[node_or_edge]
#                         elif node_or_edge in graph.nodes: # single path node
#                             # TODO: add a dependency on the terminality of the source (see equation defined in class docstring)!
#                             val = -1 * self.freewheel_path_removal_coeff * free_wheeling_node_scores[node_or_edge]
#                     elif op in configuration.REDUCTION_EVO_OPERATORS.values() and node_or_edge in graph.nodes:
#                         val = max(-1 * self.freewheel_removal_coeff * free_wheeling_node_scores[node_or_edge],
#                                 -1 * self.terminal_removal_coeff * terminal_node_scores[node_or_edge])
#                     else: # else case covers all operators on edges, and any growth/swap operators
#                         val = self.default_val
#                 else:
#                     val = -1 * np.inf
#                 value_matrix.set_prob_by_nodeEdge_op(val, node_or_edge, op)

#         return value_matrix
    
#     @staticmethod
#     def _is_multipath_node(graph, node_or_edge):
#         if node_or_edge in graph.nodes and (graph.get_in_degree(node_or_edge) > 1 or graph.get_out_degree(node_or_edge) > 1):
#             return True
#         return False


class CrossoverMaker(object):
    """
    An explicit class is not really necessary right now, but might become useful if we define more crossover operators
    """
    def __init__(self, verbose=False, **attr):
        self.verbose = verbose
    
    def crossover_graphs(self, graph0, graph1):
        # check if each evolution operator is possible
        verification = [cross_op().verify_evolution(graph0, graph1) for (_, cross_op) in configuration.CROSSOVER_OPERATORS.items()]

        # choose one evolution from all possible
        possible_cross_ops = [cross_op for (verify, cross_op) in zip(verification, configuration.CROSSOVER_OPERATORS.values()) if verify]
        if len(possible_cross_ops) == 0:
            raise RuntimeError('No valid crossover operators')

        cross_op_choice = np.random.choice(possible_cross_ops)

        # apply the chosen evolution
        child0, child1 = cross_op_choice().apply_evolution(graph0, graph1, verbose=self.verbose)

        # maybe run hessian analysis here, maybe we can do something with it, maybe not (could have two classes)
        return child0, child1, cross_op_choice