#import json
import numpy as np
from numpy import pi
#from components.component_parameters import component_parameters
#from simulation_parameters import simulation_parameters

class Component(object):
    """   
    A general class for any component, to ensure ease of use in the genetic algorithms.
    """
    def __init__(self, component_parameters, component_type, component_number):
        self.component_type = component_type
        self.component_number = component_number
        self.name = component_type + str(component_number)
        
        properties = component_parameters[self.name]
        for i_property in properties: 
            setattr(self, i_property, properties[i_property])
        return
    
    
    
## ***************************************************************88    
    
class Environment(object):
    """   
    A class that stores all our environment parameters etc.
    """
    def __init__(self):
        return
    
    def reset(self):
        return


## ***************************************************************88    


class Simulator(object):
    """   
    A class that simulates a given experiment.
    """
    def __init__(self):
        return
    
    def simulate_experiment(self, individual, experiment, env):
        return
    
    def reset(self):
        return


## ***************************************************************88    

class GeneticAlgorithmParameters(object):
    """
    A simple class that stores all the common parameters for a genetic algorithm run.
    This can (ideally) be used for the outer (create experiment) GA and the inner (find best parameters) GA
    """
    def __init__(self, N_ATTRIBUTES, BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS):
        
        self.N_ATTRIBUTES = N_ATTRIBUTES
        self.BOUNDSLOWER = BOUNDSLOWER
        self.BOUNDSUPPER = BOUNDSUPPER
        self.DTYPES = DTYPES
        self.DSCRTVALS = DSCRTVALS
        
        self.N_POPULATION = 100
        self.N_GEN = 100
        self.MUT_PRB = 0.1
        self.CRX_PRB = 0.1
        self.N_HOF = 1
        self.VERBOSE = 0
        
        return

    
    
    
    
    
    
    
    