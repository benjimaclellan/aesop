"""
Benjamin MacLellan, 2020
"""

import multiprocess as mp

# ensure all components are imported here, as we will build the 'library' with them
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, DelayLine

class Config(object):
    def __init__(self):
        self.TYPE = "topology optimization hyperparameters"
        self.NFITNESS = 1  # how many values to optimize
        self.WEIGHTS = (-1.0,)  # weights to put on the multiple fitness values
        self.MULTIPROC = False  # multiprocess or not
        self.NCORES = mp.cpu_count()  # number of cores to run multiprocessing with
        self.N_POPULATION = 3  # number of individuals in a population
        self.N_GEN = 5 # number of generations
        self.MUT_PRB = 1.0  # independent probability of mutation
        self.CRX_PRB = 0.0  # independent probability of cross-over
        self.N_HOF = 1  # number of inds in Hall of Fame (num to keep)
        self.VERBOSE = 1  # verbose print statement for GA statistics
        self.INIT = None
        self.NUM_ELITE = 1
        self.NUM_MATE_POOL = self.N_POPULATION - self.NUM_ELITE
        self.library = {'splitters': [PowerSplitter],
                        'nonsplitters': [Fiber, PhaseModulator, WaveShaper]}
        self.CALLBACK = None
        return