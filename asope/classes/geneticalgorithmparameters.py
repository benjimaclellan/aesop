
"""
GeneticAlgorithmParameters simply contains variables used in the genetic algorithm, but no class functions. 
"""    

#%%
class GeneticAlgorithmParameters(object):
    """
        A simple class that stores all the common parameters for running the genetic algorithm.
    """
    def __init__(self):
        self.N_POPULATION = 100
        self.N_GEN = 100
        self.MUT_PRB = 0.1
        self.CRX_PRB = 0.1
        self.N_HOF = 1
        self.VERBOSE = 0
        return