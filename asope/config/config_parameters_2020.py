"""
example config class - to create a new config, copy the same structure
"""

import multiprocess as mp

# ensure all components are imported here, as we will build the 'library' with them
from classes.components import Fiber, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, DelayLine

class Config(object):
    def __init__(self):
        self.TYPE = "parameter"
        self.NFITNESS = 1  # how many values to optimize
        self.WEIGHTS = (1.0,)  # weights to put on the multiple fitness values
        self.MULTIPROC = False  # multiprocess or not
        self.NCORES = mp.cpu_count()  # number of cores to run multiprocessing with
        self.N_POPULATION = 25  # number of individuals in a population (make this a multiple of NCORES!)
        self.N_GEN = 10  # number of generations
        self.MUT_PRB = 0.01  # independent probability of mutation
        self.CRX_PRB = 0.6  # independent probability of cross-over
        self.N_HOF = 1  # number of inds in Hall of Fame (num to keep)
        self.VERBOSE = True  # verbose print statement for GA statistics
        self.INIT = None
        self.GRADIENT_DESCENT = 'numerical'
        self.FINE_TUNE = True
        self.ALPHA = 0.00005
        self.MAX_STEPS = 2000
        self.NUM_ELITE = 2
        self.NUM_MATE_POOL = self.N_POPULATION // 2 - self.NUM_ELITE
        return