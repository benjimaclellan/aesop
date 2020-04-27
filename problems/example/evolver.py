import numpy as np
import copy

from .evolution_operators.evolution_operators import *
import config.config as configuration


class Evolver(object):

    def __init__(self, **attr):
        super().__init__(**attr)
        return


    def evolve_graph(self, graph, evaluator):
        """Function
        """

        # check if each evolution operator is possible

        verification = [evo_op().verify_evolution(graph) for (_, evo_op) in configuration.EVOLUTION_OPERATORS.items()]

        possible_evo_ops = [evo_op for (verify, evo_op) in zip(verification, configuration.EVOLUTION_OPERATORS.values()) if verify]
        evo_op_choice = np.random.choice(possible_evo_ops)

        graph = evo_op_choice().apply_evolution(graph)

        # maybe run hessian analysis here, maybe we can do something with it, maybe not (could have two classes)
        # score = evaluator.evaluate_graph(graph)


        return graph

