import autograd.numpy as np
import random
import matplotlib.pyplot as plt
import pickle

from .evolution_operators.evolution_operators import *
import config.config as configuration


class Evolver(object):
    """
    """
    def __init__(self, verbose=False, operator_likelihood=None, **attr):
        """
        :param operator_likelihood: likelihood weighting the probability of each evolution operator being applied. If None, probability of each is equal
                                    These likelihoods are not required to be probabilities: they are rescaled, but their relative values matter
                                    operator_likelihood must be provided as a dictionary with (key, val) = (operator_class_name, likelihood)
                                    likelihood can be a float, or a scalar function of the current generation, but the type must be consistent for each operator name (i.e. all floats, or all funcs)
        """
        self.verbose = verbose
        self.time_dependent_likelihood = False
        if operator_likelihood is None:
            self.operator_likelihood = [1] *  len(configuration.EVOLUTION_OPERATORS)
        elif callable(operator_likelihood.values()[0]):
            self.time_dependent_likelihood = True
            # maps generation to an array of the likelihood at time gen
            self.operator_likelihood = lambda gen :[operator_likelihood[operator.__class__.__name__](gen) for operator in configuration.EVOLUTION_OPERATORS.values()]
        else:
            self.operator_likelihood = [operator_likelihood[operator.__class__.__name__] for operator in configuration.EVOLUTION_OPERATORS.values()]

        super().__init__(**attr)
        return


    def evolve_graph(self, graph, evaluator, generation=None, verbose=False):
        """
        Function
        """
        # check if each evolution operator is possible
        verification = [evo_op().verify_evolution(graph) for (_, evo_op) in configuration.EVOLUTION_OPERATORS.items()]

        # choose one evolution from all possible
        possible_evo_ops = [evo_op for (verify, evo_op) in zip(verification, configuration.EVOLUTION_OPERATORS.values()) if verify]

        if self.time_dependent_likelihood:
            if generation is None:
                raise ValueError('Generation must be provided for time dependent likelihoods')
            probability = np.array([prob for (prob, verify) in zip(self.operator_likelihood(generation), verification) if verify])
        else:
            probability = np.array([prob for (prob, verify) in zip(self.operator_likelihood, verification) if verify])

        probability = probability / np.sum(probability)
        evo_op_choice = np.random.choice(possible_evo_ops, p=probability)

        # apply the chosen evolution
        graph = evo_op_choice().apply_evolution(graph, verbose=self.verbose)

        # maybe run hessian analysis here, maybe we can do something with it, maybe not (could have two classes)
        return graph, evo_op_choice

    def random_graph(self, graph, evaluator):
        N_EVOLUTIONS = 10
        for n in range(N_EVOLUTIONS):
            try:
                graph_tmp, evo_op = self.evolve_graph(graph, evaluator)
                graph_tmp.assert_number_of_edges()
                graph = graph_tmp
            except:
                continue

        return graph


class StochMatrixEvolver(object):
    """
    Evolver object will take in and update graphs based on a stochastic probability matrix
    It will have a set of rules for updating a graph's stochastic matrix, in order to update the graph next time

    Matrices are defined as a probability matrix of Node/Edge versus Evolution Operator. p(node0, SwapNode) for example is the probability
    that the 'SwapNode' operator be applied on node node0. 

    The application of any operator must take the form apply_evolution_at(graph, node_or_edge, verbose=whatever)

    This class has the most basic probability selection: if a given operator can possibly be run on a node/edge, it will be assigned a value of one.
    If it is not possible, it is assigned a value of zero. Probability are normalized prior to selecting the operator
    """
    def __init__(self, verbose=False, **attr):
        self.verbose = verbose
        self.evo_op_list = list(configuration.EVOLUTION_OPERATORS.values()) # we pick these out because technically dictionary values are not ordered
                                                                            # so because we need our matrix order to be consistent, 
        super().__init__(**attr)
    
    def evolve_graph(self, graph, evaluator, generation=None, verbose=False, debug=False):
        """
        Evolves graph according to the stochastic matrix probabilites

        :param graph: the graph to evolve
        :param evaluator: not used in the base implementation, but may be useful in the future
        """
        if graph.evo_probabilities_matrix is None:
            self.create_graph_matrix(graph, evaluator)
        
        if debug:
            print(f'evolution probability matrix for graph {graph}')
            print(graph.evo_probabilities_matrix)
            print()

        node_or_edge, evo_op = graph.evo_probabilities_matrix.sample_matrix()
        graph = evo_op().apply_evolution_at(graph, node_or_edge, verbose=verbose)

        self.update_graph_matrix(graph, evaluator, evo_op, node_or_edge)
        
        return graph, evo_op

    def create_graph_matrix(self, graph, evaluator):
        """
        :param graph: the graph to evolve
        :param evaluator: not used in the base implementation, but may be useful in the future
        """
        graph.evo_probabilities_matrix = self.ProbabilityMatrix(self.evo_op_list, list(graph.nodes), list(graph.edges))
        for node_or_edge in list(graph.nodes) + list(graph.edges):
            for op in self.evo_op_list:
                likelihood = int(op().verify_evolution_at(graph, node_or_edge))
                graph.evo_probabilities_matrix.set_prob_by_nodeEdge_op(likelihood, node_or_edge, op)
        
        graph.evo_probabilities_matrix.normalize_matrix()
        graph.evo_probabilities_matrix.verify_matrix()

    def update_graph_matrix(self, graph, evaluator, last_evo_op, last_node_or_edge):
        """
        Function does not use evaluator, last_evo_op, or last_node_or_edge in this implementation. However, they may be valid parameters to modify later
        """
        self.create_graph_matrix(graph, evaluator) # just fully remakes the graph matrix for now

    def random_graph(self, graph, evaluator, n_evolutions=10, view_evo=False, verbose=False, debug=False):
        for n in range(n_evolutions):
            if verbose:
                print(f'Starting evolution number {n}')
            try:
                graph_tmp, evo_op = self.evolve_graph(graph, evaluator, generation=n, verbose=verbose, debug=debug)
                graph_tmp.assert_number_of_edges()
                graph = graph_tmp
                if view_evo:
                    graph.draw()
                    plt.show()
            except AssertionError as e:
                print(e)

        return graph
    
    def close(self):
        pass

    class ProbabilityMatrix(object):
        """
        Probability matrix of (node/edge) v. Operator. The probability p(node_or_edge, operator) is the odd of operator being applied on node_or_edge

        Note that updates to the matrix can be done either through the provided function, or by direct manipulation of self.matrix
        """
        def __init__(self, op_list, node_list, edge_list):
            self.op_to_index = {op: i for (i, op) in enumerate(op_list)}
            self.index_to_op = {i: op for (i, op) in enumerate(op_list)}
            self.node_or_edge_to_index = {node_or_edge: i for (i, node_or_edge) in enumerate(node_list + edge_list)}
            self.index_to_node_or_edge = {i: node_or_edge for (i, node_or_edge) in enumerate(node_list + edge_list)}

            self.matrix = np.ones((len(self.node_or_edge_to_index), len(self.op_to_index)))
            self.normalize_matrix()
    
        def normalize_matrix(self):
            self.matrix = self.matrix / np.sum(self.matrix)
        
        def verify_matrix(self):
            assert np.isclose(np.sum(self.matrix), 1) # matrix sums to proper probability
            assert (self.matrix <= 1).all() and (self.matrix >= 0).all() # assert all probabilities are between 0 and 1
        
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
        
        def __str__(self):
            string_rep = f'operator to index: {self.op_to_index}\n' + f'node/edge to index: {self.node_or_edge_to_index}\n' + \
                         f'matrix: \n{self.matrix}'
            return string_rep


class SizeAwareMatrixEvolver(StochMatrixEvolver):
    """
    The size aware stochastic matrix evolver updates probabilities of certain operators being used based on the size of the graph.
    The algorithm considers "growth" operators (add nodes or edges), and "reduction" operators (remove nodes or edges). Any operators not marked
    as such via decorators will be considered neutral operators, which do not affect the graph complexity (e.g. SwapNode)

    The probability rules are defined as such:
    1. If the 'verify_operator_at' methods return false, the probability of an operator at a given node/edge is zero regardless of other rules
    2. Each object has a 'ideal_node_num' parameter. This is the node number at which growth, reduction, and neutral operators have equal probability of being selected
            - This number can be updated whenever 
    3. Let's define likelihood as probability before normalization. This evolver assigns a likelihood of 1 for neutral operators, 1 - alpha to growth operators,
       and 1 + alpha to reduction operators. Alpha is clipped at +=0.9 by default, but can be clipped at any amplitude between 0 and 1 if desired
    4. alpha = alpha(delta) is an odd function (where alpha < 0 where delta < 0), where delta = <# of nodes in the graph> - <ideal # of nodes>. alpha(0) = 0
    """
    def __init__(self, verbose=False, ideal_node_num=8, alpha_func=lambda delta: (delta / 10)**3, alpha_bound=0.9, **attr):
        """
        Initializes SizeAwareMatrixEvolver (see class description above)

        :param verbose: if True, evolution operators type and node/edges are printed to stdout. Otherwise, the evolution is 'silent'
        :param ideal_node_number: number of nodes in a graph at which the growth and reduction evolution operators have equal chances of being selected
        :param alpha_func: requirements are alpha(delta) is odd, alpha(0) = 0, alpha(delta < 0) < 0. Current function is selected such that alpha(10) = 1 (max bias)
                           and such that the bias in likelihood be small if delta is small.
        :param alpha_bound: amplitude at which to clip . Also supports alpha_bound as a tuple (e.g (-0.8, 0.9)). Note that alpha_bound=A is identical to alpha_bound=(-A, A)

        Only the ideal node number is intended to be updated over the course of the object's existence (there's a useful case for it I can see, unlike the other params)
        That said, updates to any of these parameters is possible (i.e. the code will use new alpha_func and alpha_max if their values are changed)
        """
        super().__init__(verbose=verbose, **attr)
        self.ideal_node_num = ideal_node_num
        self.alpha_func = alpha_func

        # verify that the alpha_max input is valid. We'd verify for alpha_funct too but it's too complicated so
        if type(alpha_bound) is int or type(alpha_bound) is float:
            if alpha_bound < 0 or alpha_bound > 1:
                raise ValueError(f'alpha_bound given as a float/int must be in [0, 1]. {alpha_bound} is not valid')
            self.alpha_bound = (-alpha_bound, alpha_bound)
        else:
            if alpha_bound[0] > alpha_bound[1]:
                raise ValueError(f'alpha_bound upper bound {alpha_bound[1]} must be greater than alpha_bound lower bound {alpha_bound[0]}')
            elif -1 < alpha_bound[0] or alpha_bound[0] > 0:
                raise ValueError(f'Lower bound value for alpha_bound {alpha_bound[0]} must be in [-1, 0]')
            elif 0 < alpha_bound[1] or alpha_bound[1] > 1:
                raise ValueError(f'Upper bound value for alpha_bound {alpha_bound[1]} must be in [0, 1]')
            self.alpha_bound = alpha_bound

    def create_graph_matrix(self, graph, evaluator):
        """
        :param graph: the graph to evolve
        :param evaluator: not used in this implementation, but may be useful in the future
        """
        graph.evo_probabilities_matrix = self.ProbabilityMatrix(self.evo_op_list, list(graph.nodes), list(graph.edges))
        alpha = self._get_alpha_offset(graph)
        for node_or_edge in list(graph.nodes) + list(graph.edges):
            for op in self.evo_op_list:
                evo_possible = op().verify_evolution_at(graph, node_or_edge)
                if evo_possible:
                    if op.__name__ in configuration.GROWTH_EVO_OPERATORS.keys():
                        likelihood = 1 - alpha
                    elif op.__name__ in configuration.REDUCTION_EVO_OPERATORS.keys():
                        likelihood = 1 + alpha
                    else: likelihood = 1
                else: 
                    likelihood = 0
                graph.evo_probabilities_matrix.set_prob_by_nodeEdge_op(likelihood, node_or_edge, op)
        
        graph.evo_probabilities_matrix.normalize_matrix()
        graph.evo_probabilities_matrix.verify_matrix()

    def _get_alpha_offset(self, graph):
        delta = len(graph.nodes) - self.ideal_node_num
        return np.clip(self.alpha_func(delta), self.alpha_bound[0], self.alpha_bound[1])


class ReinforcementMatrixEvolver(StochMatrixEvolver):
    """
    Loosely based on reinforcement learning ideas, where only the operator selection is reinforcement based (node selection remains arbitrary)
    The "state" of the graph is solely defined by the number of nodes, for simplicity

    1. The probability matrix is arbitrary if we have no past history for a graph of size N (number of nodes)
    2. Otherwise, the best historical solution op is assigned probability 1 - epsilon (equally divided between all possible node/edges),
       with equal probability across all other possible node/op pairs (epsilon can be gen dependent)
    
    Implementation notes:
    1. Each graph can store its previous scores / averages? At the beginning of each new optimization we can compute the updated "value"
    2. Consider: how do we backpropagate value between states. For now we won't, we just calculate improvement in score from last time

    The philosophy for now is: immediate improvement is the name of the game. We define the value of an operation as its predicted ,
    where reward is improvement in score. The predicted improvement is aggregated from all graphs that the evolver is called on

    # TODO: consider making value less based on immediate reward
    # TODO: consider decaying epsilon scheme instead of constant epsilon value
    """
    def __init__(self, verbose=False, starting_value_matrix=None, epsilon=0.4, **attr):
        """
        Creates a ReinforcementMatrixEvolver (reinforcement is used a bit loosely here)

        :param verbose: if True, evolution operators type and node/edges are printed to stdout. Otherwise, the evolution is 'silent'
        :param starting_value_matrix: initial matrix of expected awards for different operators
                                      Parameter is a tuple: (dictionary of numpy 2D vectors, ordered list of operators)
                                      Dictionary key: number of nodes in graph, N
                                      Dictionary value: row1 = numpy array of expected rewards (i.e. values) per operator (in same order as the ordered list)
                                                        row2 = numpy array of integers, with number of times the same-index value of expected rewards has been updated in the past (required to tally long-term average)
                                      If not provided, all configuration operators are selected, and all values are set to 0 to start

                                      If a string, this is the string to a pickle file of a previous run
        """
        super().__init__(verbose=verbose, **attr)
        self.epsilon = epsilon
        
        if starting_value_matrix is None:
            self.value_matrix = {} # value matrix will be consistently updated
        elif type(starting_value_matrix) == str:
            with open(starting_value_matrix, 'rb') as handle:
                self.value_matrix = pickle.load(handle)
        else:
            self.value_matrix = starting_value_matrix[0]
            self.evo_op_list = starting_value_matrix[1] # override default in constructor
    
    def create_graph_matrix(self, graph, evaluator):
        self._update_value_matrix(graph)
        self._translate_values_to_probabilities(graph, evaluator)

    def _update_value_matrix(self, graph):
        if len(graph.nodes) not in self.value_matrix: # check whether we already have an entry for this graph state
            self.value_matrix[len(graph.nodes)] = np.zeros((2, len(self.evo_op_list)))

        if graph.last_evo_record is not None and graph.score is not None and graph.last_evo_record['score'] is not None: # we can only update our evo matrices if we have a previous score to compare to
            N = graph.last_evo_record['state']
            reward = graph.last_evo_record['score'] - graph.score
            evo_index = self.evo_op_list.index(graph.last_evo_record['op'])

            # new value is the new average reward. If the previous average is Qk, and the new reward is Rk:
            # average reward Q(k + 1) = Qk + (1/k)(R_k - Q_k)
            self.value_matrix[N][1][evo_index] += 1
            Qk = self.value_matrix[N][0][evo_index]
            k = self.value_matrix[N][1][evo_index]
            new_value = Qk + (1 / k) * (reward - Qk)

            self.value_matrix[N][0][evo_index] = new_value

    
    def _translate_values_to_probabilities(self, graph, evaluator):
        """
        Turns the value matrix into a matrix of probabilities with the following rules:
        1. Total probability of the greedy operator being selected is 1 - epsilon
        2. Probability of greedy operator is evenly dispersed among each node/edge where it can be applied
        3. Remaining probability is distributed equally among all other possible operator-node/edge combinations

        Pre-condition: assumes self.value_matrix is fully updated
        """
        super().create_graph_matrix(graph, evaluator) # get basic matrix with 1 for possible combos and 0 for impossible combos

        greedy_op, greedy_op_num_possible = self._get_greedy_op_and_possible_num(graph)
        greedy_op_prob = (1 - self.epsilon) / greedy_op_num_possible
        other_op_prob = self.epsilon / (np.count_nonzero(graph.evo_probabilities_matrix.matrix) - greedy_op_num_possible)
        
        for node_or_edge in list(graph.nodes) + list(graph.edges):
            for op in self.evo_op_list:
                if not np.isclose(0, graph.evo_probabilities_matrix.get_prob_by_nodeEdge_op(node_or_edge, op)): # i.e. if an operation is possible
                    prob = greedy_op_prob if op == greedy_op else other_op_prob
                    graph.evo_probabilities_matrix.set_prob_by_nodeEdge_op(prob, node_or_edge, op)

        # omit normalization because evo matrix should be normalized already. If not, we want it to crash, not silently fail
        graph.evo_probabilities_matrix.verify_matrix()

    def _get_greedy_op_and_possible_num(self, graph):
        """
        Returns the greedy operator. If there exist multiple potential greedy operators, it randomly selects one
        """
        # TODO: refactor this, it's ugly code...
        # 1. Look through all max valued options, going in descending order of value (in case the max valued options are not applicable)
        usable = np.ones_like(self.value_matrix[len(graph.nodes)][0])
        while np.count_nonzero(usable) > 0:
            max_val = np.max(np.array([self.value_matrix[len(graph.nodes)][0][i] for i in range(len(usable)) if usable[i]]))
            indices = [i for (i, val) in enumerate(self.value_matrix[len(graph.nodes)][0]) if np.isclose(val, max_val)]
            while len(indices) > 0:
                index = np.random.choice(indices)
                op = self.evo_op_list[index]
                greedy_op_num_possible = np.count_nonzero(graph.evo_probabilities_matrix.get_probs_at_operator(op))
                if greedy_op_num_possible > 0:
                    return op, greedy_op_num_possible
                else:
                    indices.remove(index)
            
            for i in range(len(usable)):
                if np.isclose(max_val, self.value_matrix[len(graph.nodes)][0][i]):
                    usable[i] = 0

        # 2. no max valued option is available
        raise ValueError('No possible operator found')

    def evolve_graph(self, graph, evaluator, generation=None, verbose=False, debug=False):
        """
        Evolves graph according to the stochastic matrix probabilites
        The probabilities for operators are based on a epsilon-greedy learning method
        The probabilities for different nodes with the same operator are equal, assuming the operation is possible

        :param graph: the graph to evolve
        :param evaluator: not used in the base implementation, but may be useful in the future
        """
        pre_evo_state = len(graph.nodes)
        graph, evo_op = super().evolve_graph(graph, evaluator, generation=generation, verbose=verbose, debug=debug)
        if graph.last_evo_record is None:
            graph.last_evo_record = {}
        graph.last_evo_record['op'] = evo_op
        graph.last_evo_record['score'] = graph.score
        graph.last_evo_record['state'] = pre_evo_state

        return graph, evo_op
    
    def close(self):
        with open(f'reinforcement_evolver_value_matrix.pkl', 'wb') as handle:
            pickle.dump(self.value_matrix, handle)


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