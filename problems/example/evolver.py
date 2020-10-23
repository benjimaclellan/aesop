import autograd.numpy as np
import random
import matplotlib.pyplot as plt

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
            # TODO: add warning if we're trying to give a likelihood to an operator that does not exist
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
                graph_tmp = self.evolve_graph(graph, evaluator)
                graph_tmp.assert_number_of_edges()
                graph = graph_tmp
            except:
                continue

        return graph

class StochMatrixEvolver(object):
    """
    Evolver object will take in and update graphs based on a stochastic probability matrix
    It will have a set of rules for updating a graph's stochastic matrix, in order to update the graph next time

    Matrices are defined as a probability matrix of Node/Edge versus Evolution Operator. p(node0, SwapNode) for example is the relative probability
    that the 'SwapNode' operator be applied on node node0

    The application of any operator must take the form apply_evolution_at(graph, node_or_edge, verbose=whatever)

    This class has the most basic probability selection: if a given operator can possibly be run on a node/edge, it will be assigned a value of one.
    If it is not possible, it is assigned a value of zero. Probability are normalized prior to selecting the operator
    """
    def __init__(self, verbose=False, **attr):
        self.verbose = verbose
        self.evo_op_list = list(configuration.EVOLUTION_OPERATORS.values()) # we pick these out because technically dictionary values are not ordered
                                                                            # so because we need our matrix order to be consistent, 
        super().__init__(**attr)
    
    def evolve_graph(self, graph, evaluator, generation=None, verbose=False):
        """
        Evolves graph according to the stochastic matrix probabilites

        :param graph: the graph to evolve
        :param evaluator: not used in the base implementation, but may be useful in the future
        """
        if graph.evo_probabilities_matrix is None:
            self.create_graph_matrix(graph, evaluator)
        
        print(f'graph matrix: {graph.evo_probabilities_matrix}')
        node_or_edge, evo_op = graph.evo_probabilities_matrix.sample_matrix()
        graph = evo_op().apply_evolution_at(graph, node_or_edge, verbose=verbose)
        self.update_graph_matrix(graph, evaluator, evo_op, node_or_edge)
        return graph

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

    def random_graph(self, graph, evaluator, view_evo=False, verbose=False):
        N_EVOLUTIONS = 10
        for n in range(N_EVOLUTIONS):
            try:

                graph_tmp = self.evolve_graph(graph, evaluator, generation=n, verbose=verbose)
                graph_tmp.assert_number_of_edges()
                graph = graph_tmp
                if view_evo:
                    graph.draw()
                    plt.show()
            except AssertionError as e:
                print(e)

        return graph


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
            self.matrix[self.node_or_edge_to_index[node_or_edge]][self.op_to_index[op]] = probability
        
        def __str__(self):
            string_rep = f'operator to index: {self.op_to_index}\n' + f'node/edge to index: {self.node_or_edge_to_index}\n' + \
                         f'matrix: \n{self.matrix}'
            return string_rep


class RuleBasedEvolver(Evolver):
    """
    Evolves according to a set of rules with a certain probability, and randomly with another probability

    The trend is that the exploitativeness of the algorithm will increase over time, as exploration decreases:
    https://towardsdatascience.com/striking-a-balance-between-exploring-and-exploiting-5475d9c1e66e
    https://www.manifold.ai/exploration-vs-exploitation-in-reinforcement-learning

    We assume that the decay in exploitation is linear, for simplicity
    """
    def __init__(self, epsilon_start=0.6, epsilon_end=0.1, decay_length_gen=10, verbose=False, operator_likelihood=None, **attr):
        """
        Rule based evolver evolves randomly with probability epsilon, and based on the rules with probability 1 - epsilon
        """
        def epsilon_from_gen(gen):
            if gen > decay_length_gen:
                return epsilon_end
            else:
                return (epsilon_end - epsilon_start) / decay_length_gen * gen + epsilon_start

        self.epsilon_from_gen = epsilon_from_gen

        super().__init__(verbose=verbose, operator_likelihood=operator_likelihood, **attr)
    
    def evolve_graph(self, graph, evaluator, generation=None, verbose=False):
        if random.random() < self.epsilon_from_gen(generation):
            return super().evolve_graph(graph, evaluator, generation=generation, verbose=verbose)
        
        evo_op_choice = self.select_correct_evolution(graph)
        # apply the chosen evolution
        graph = evo_op_choice().apply_evolution(graph, verbose=self.verbose)

        # maybe run hessian analysis here, maybe we can do something with it, maybe not (could have two classes)
        return graph, evo_op_choice
    
    def select_correct_evolution(self, graph):
        # TODO: have some selection rule! this is currently here as a stopgap
        verification = [evo_op().verify_evolution(graph) for (_, evo_op) in configuration.EVOLUTION_OPERATORS.items()]
        possible_evo_ops = [evo_op for (verify, evo_op) in zip(verification, configuration.EVOLUTION_OPERATORS.values()) if verify]
        return np.random.choice(possible_evo_ops)


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