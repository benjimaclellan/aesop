import re
import fnmatch
import json
import numpy as np

from beeprint import pp 

#import random 
from classes import Component

def buildexperiment(component_parameters, experiment_nums=None):
    ## Get info about our possible components
    components_names = np.array(list(component_parameters.keys()))
    
    ## The most components we can have in the setup
    maxnum_components = 3
    
    ## Here, we make a list of our components that make up the experiment
    if experiment_nums == None:
        experiment_nums = np.random.randint(0, len(components_names), maxnum_components)
    
    experiment_components = components_names[ experiment_nums ]
    
    ## Ensure we don't have redundant components (same fiber adjacent to itself)
    experiment_components = checkcomponents(experiment_components)
    
    ## Build a list of objects for each component, aka our experiment on the optical table
    experiment = list()
    for i_component in range(len(experiment_components)):
        (i_type, i_num) = splitstring(experiment_components[i_component])
        experiment.append( Component(component_parameters, i_type, i_num) )

    return experiment



def extract_bounds(experiment):
    N_ATTRIBUTES = 0
    BOUNDSUPPER = []
    BOUNDSLOWER = [] 
    DTYPES = [] 
    DSCRTVALS = []
    for component in experiment:
        N_ATTRIBUTES += component.N_PARAMETERS
        BOUNDSLOWER += component.LOWER
        BOUNDSUPPER += component.UPPER
        DTYPES += component.DTYPE
        DSCRTVALS += component.DSCRTVAL
    assert(N_ATTRIBUTES == len(BOUNDSLOWER) == len(BOUNDSUPPER) == len(DTYPES))    
    
    return N_ATTRIBUTES, BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS



def experiment_description(experiment, verbose=False, individual=None):
    """
    Prints a summary of the experiment setup
    """
    print('****Experiment description****')
    i = 0
    j = 0
    for component in experiment:
        print('Component {:d}: {:s}'.format(i, component.name))
        if verbose == True:
            pp(component)
        if individual is not None:
            print('Attributes for this component are: {}'.format(individual[j:j+component.N_PARAMETERS]))
        print('\n')
        i += 1
        j += component.N_PARAMETERS
            




def splitstring(string):
    """   
    Given a string, i.e. 'fiber1', will split into string and numeric parts, i.e. 'fiber' and 1 (as an integer)
    """
    match = re.match(r"([a-z]+)([0-9]+)", string, re.I)
    if match:
        (characters, num) = match.groups()
    return (characters, int(num))    




def checkcomponents(exp_comps):
    """
    Ensure adjacent components are not both the same fiber or same waveshapers, removing redundancy in the experiment and (potentially?) leading to a more stable optimization. We don't check if adjacent components are identical phase modulators as this is valid
    """
    
    rm_comps = list()
    
    for i in range( len(exp_comps) - 1 ):
        if exp_comps[i] == exp_comps[i+1]:
            
            ## Check if adjacent fibers are the same kind
            check_fiber = bool(fnmatch.fnmatch(exp_comps[i], 'fiber*'))
            if check_fiber:
                rm_comps.append(i)
            
            ## Check if adjacent waveshapers are the same kind
            check_waveshaper = bool(fnmatch.fnmatch(exp_comps[i], 'awg*'))
            if check_waveshaper:
                rm_comps.append(i)
            
            ## Check if adjacent waveshapers are the same kind
            check_waveshaper = bool(fnmatch.fnmatch(exp_comps[i], 'waveshaper*'))
            if check_waveshaper:
                rm_comps.append(i)
                
    ## Create our redundancy-free list of components in the experiment
    checkedexp_comps = []
    for i in range( len(exp_comps) ):
        if i not in rm_comps:
            checkedexp_comps.append( exp_comps[i] )
        
    return checkedexp_comps



def savesetup(experiment):
    """
    Saves a given experiment to a JSON file, which can be used later or read by a human
    """
    with open('results/results.json', 'w') as outfile:
        json.dump(experiment, outfile)
    pass


def savelogbook(logbook, filepath):
    import pandas as pd
    df_log = pd.DataFrame(logbook)
    df_log.to_csv(filepath + '.csv', index=False)
    
    
def extractlogbook(logbook):
   hrlog = list(logbook[0].keys())
   log = {}
   
   for stat in hrlog:
       log[stat] = [item[stat] for item in logbook] 
    
   return log