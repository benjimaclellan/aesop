import autograd.numpy as np
import random

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