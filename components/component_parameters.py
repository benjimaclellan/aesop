from math import pi

"""
Component Definitions:
    Here, every possible component we wish to consider is described.
    Ensure that each component follows the same structure.
    UPPER and LOWER *must* be lists, in order to build a list of parameters to be submitted to the inner GA
"""
## -----------------------------
def component_parameters(component_id):

    if component_id == 0:
        cp = {  "type":"fiber",
                "typeval":"0",  
            		"beta": [2e-20],
            	 	"Tr": 0.5,
            	 	"gamma": 100,
            	 	"N_PARAMETERS": 1,
            	 	"UPPER": [20000],
            	 	"LOWER": [0],
                "DTYPE": ['float'],
                "DSCRTVAL": [None],
                "FINETUNE_SKIP":0
         	}
        
    elif component_id == 1:
        cp = {  "type":"awg",
                "typeval":"0",  
                "N_PARAMETERS": 2,
            	 	"UPPER": [8] + [2*pi],   # phase bins
            	 	"LOWER": [1] + [0],
                "DTYPE": ['int', 'float'],
                "DSCRTVAL": [1, None],
                "FINETUNE_SKIP":1 
                }
        
    elif component_id == 2:
        cp = {  "type":"phasemodulator",
                "typeval":"0",  
                "vpi": 1.0,
                "N_PARAMETERS": 3,
                "HPBoundLOW": 0,
                "HPBoundUP": 1,
            	 	"UPPER": [10, 200e6, 2*pi],   # max shift, frequency, phase offset
            	 	"LOWER": [0, 1e6, 0],
                "DTYPE": ['float', 'float', 'float'],
                "DSCRTVAL": [None, 1e6, None],
                "FINETUNE_SKIP":0
             	}
    else:
        print(component_id)
        raise ValueError('No component with this ID')
        
    return cp


