from numpy import pi
import numpy as np


"""
Component Definitions:
    Here, every possible component we wish to consider is described.
    Ensure that each component follows the same structure.
    UPPER and LOWER *must* be lists, in order to build a list of parameters to be submitted to the inner GA
"""
def component_parameters(env, sim):
#    n_bins_awg = 10
    n_bins_awg = int(max([sim.FitnessFunctions.q/sim.FitnessFunctions.p,sim.FitnessFunctions.p/sim.FitnessFunctions.q]))

    
    
    
    n_bins_ws = 10
    
    """
    
    """
    component_parameters = {
            
        ## Component 0
    	"fiber0":{
    		"beta": [2e-20],
    	 	"Tr": 0.5,
    	 	"gamma": 100,
    	 	"N_PARAMETERS": 1,
    	 	"UPPER": [10000],
    	 	"LOWER": [0],
        "DTYPE": ['float'],
        "DSCRTVAL": [None]
     	}
        ,
        
        ## Component 1
     	"fiber1":{
    		"beta": [-1.5e-20],
    	 	"Tr": 0.1,
    	 	"gamma": 70,
    	 	"N_PARAMETERS": 1,
    	 	"UPPER": [3000],
    	 	"LOWER": [0],
        "DTYPE": ['float'],
        "DSCRTVAL": [None]
     	}
         ,
        
#        ## Component 2
#     	"phasemodulator0":{
#    		"vpi": 1.0,
#        "N_PARAMETERS": 3,
#    	 	"UPPER": [10, 200, 2*pi],   # max shift, frequency, phase offset
#    	 	"LOWER": [0, 1, 0],
#        "DTYPE": ['float', 'float', 'float'],
#        "DSCRTVAL": [None, 1, None]
#     	}
#        ,
        ## Component 2
     	"phasemodulator0":{
    		"vpi": 1.0,
        "N_PARAMETERS": 3,
    	 	"UPPER": [10, 10, 2*pi],   # max shift, frequency, phase offset
    	 	"LOWER": [0, 1, 0],
        "DTYPE": ['float', 'float', 'float'],
        "DSCRTVAL": [None, 1, None]
     	}
        ,
        
        # Component 3
     	"waveshaper0":{
    		"totalbandwidth": 5.0,
    		"slicewidth": 12.5e9,
        "N_PARAMETERS": 2 * n_bins_ws,
        "N_SLICES": n_bins_ws,
    	 	"UPPER": n_bins_ws*[2*pi] + n_bins_ws*[1],
    	 	"LOWER": n_bins_ws*[0] + n_bins_ws*[0.8],
        "DTYPE": n_bins_ws*['float'] + n_bins_ws*['float'],
        "DSCRTVAL": n_bins_ws*[pi/2] + n_bins_ws*[None]
     	}
        ,
        
#        ## Component 4
#        "awg0":{
#        "N_PARAMETERS": 1 + n_bins_awg,
#    	 	"UPPER": [n_bins_awg] + n_bins_awg*[2*pi],   # phase bins
#    	 	"LOWER": [1] + n_bins_awg*[0],
#        "DTYPE": ['int'] + n_bins_awg*['float'],
#        "DSCRTVAL": [1] + n_bins_awg*[None]
#     	}
#         ,
         
        ## Component 4
        "awg0":{
        "N_PARAMETERS": n_bins_awg,
    	 	"UPPER": n_bins_awg*[2*pi],   # phase bins
    	 	"LOWER": n_bins_awg*[0],
        "DTYPE": n_bins_awg*['float'],
        "DSCRTVAL": n_bins_awg*[None]
     	}
         ,
    
    }
    return component_parameters