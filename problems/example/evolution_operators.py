from lib.decorators import register_evolution_operators
from lib.base_classes import EvolutionOperators as EvolutionOperatorsParent



class EvolutionOperators(EvolutionOperatorsParent):

    def __init__(self, **attr):
        super().__init__(**attr)
        return


    @staticmethod
    @register_evolution_operators
    def add_node():
        """Function.
        """
        print('inside add_node() now')
        return

    # @staticmethod
    # @register_evolution_operators
    # def remove_node():
    #     """Function.
    #     """
    #     return

