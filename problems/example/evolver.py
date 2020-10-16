import autograd.numpy as np 

from .evolution_operators.evolution_operators import *
import config.config as configuration


class Evolver(object):
    """
    """
    def __init__(self, verbose=False, **attr):
        self.verbose = verbose
        super().__init__(**attr)
        return


    def evolve_graph(self, graph, evaluator, verbose=False):
        """
        Function
        """

        # check if each evolution operator is possible
        verification = [evo_op().verify_evolution(graph) for (_, evo_op) in configuration.EVOLUTION_OPERATORS.items()]

        # choose one evolution from all possible
        possible_evo_ops = [evo_op for (verify, evo_op) in zip(verification, configuration.EVOLUTION_OPERATORS.values()) if verify]
        evo_op_choice = np.random.choice(possible_evo_ops)

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


class CrossoverMaker():
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