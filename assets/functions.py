import fnmatch
import pickle
from beeprint import pp 
import matplotlib.pyplot as plt
import numpy as np

def P(At):
    return np.power( np.abs( At ), 2)
    
def PSD(Af, df):
    return np.power( np.abs( Af ), 2) / df

def FFT( At, dt):   # proper Fast Fourier Transform
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(At)))*dt

def IFFT( Af, dt):  # proper Inverse Fast Fourier Transform
    return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Af)))/dt

def RFSpectrum( At, dt):
    return np.abs(np.fft.rfft(np.power( np.abs( At ), 2)))







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




"""
This is a recursive function which, based on certain rules of causality/etc finds the proper order to apply the components in, on a given graph E 
"""

def recurse(E, node, out_edges, path, startpoints, in_edges):
    path_i = []    
    while True:
        path_i.append(node)
        if len(E.suc(node)) == 0:
            break        
        
        if not E.nodes[E.suc(node)[0]]['info'].splitter:
            node = E.suc(node)[0]
        else:
            break
    
    path.append(path_i)
    
    if len(E.suc(node)) == 0:
#        path.append([node])
        if len(out_edges) > 0: # check if there's more unfinished paths
            node = out_edges[0]; out_edges.pop(0)
            (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
        if len(startpoints) != 0:
            node = startpoints[0]; startpoints.pop(0)
            (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
    
    else:
        node = E.suc(node)[0]
#        path.append(path_i)
        
        if in_edges[node] == 0:# if this is the first time we encounter this node 
            for s in E.suc(node): out_edges.append(s)
            
        in_edges[node] += 1 # we've encountered this node one more time
        
        if in_edges[node] == len(E.pre(node)): # in we've run into this node the right number of times
            path.append([node]) # we can now process it
            
            if len(out_edges) > 0: # check if there's more unfinished paths
                node = out_edges[0]
                out_edges.pop(0)
                (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
                
        elif in_edges[node] != len(E.pre(node)): ## finished loop?
            if len(out_edges) > 0:
                node = out_edges[0]
                out_edges.pop(0)
                (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
            
        if len(startpoints) != 0:
            node = startpoints[0]
            startpoints.pop(0)
            (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
    
    return (E, node, out_edges, path, startpoints, in_edges)


