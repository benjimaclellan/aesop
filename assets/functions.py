import fnmatch
import pickle
from beeprint import pp 
import matplotlib.pyplot as plt

def save_experiment(filename, experiment, env):
    with open(filename+'.pkl', 'wb') as output:
        pickle.dump(experiment, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(env, output, pickle.HIGHEST_PROTOCOL)
        
def load_experiment(filename):
    with open(filename+'.pkl', 'rb') as input:
        experiment = pickle.load(input)
        env = pickle.load(input)
    return experiment, env

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


"""
Plots the temporal and spectral power for a given individual
"""
def plot_individual(env, fitness):    
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), dpi=80)
    
    ax[0].set_xlabel('Time (ps)')
    ax[0].set_ylabel('Power [arb]')
    ax[1].set_xlabel('Frequency (THz)')
    ax[1].set_ylabel('PSD [arb]')
    
    ax[0].plot(env.t/1e-12, env.P(env.At0), label='initial')
    ax[1].plot(env.f/1e12, env.PSD(env.Af0, env.df))

    
    ax[0].plot(env.t/1e-12, env.P(env.At), label='final')
    ax[1].plot(env.f/1e12, env.PSD(env.Af, env.df))

    ax[0].set_title('Fitness {}'.format(fitness))
    ax[0].legend()
    
    return fig, ax


def splitindices(num, div):
    indices = [0]
    for i in range(div):
        val = num//(div - i)
        num += -val
        prev = indices[-1]
        indices.append(val + prev)
    return indices