import pickle
import numpy as np

"""
    A number of useful functions that are utilized through the package.
"""


def P(At):
    """
        Power of a signal, as the square of the absolute value
    """
    return np.power( np.abs( At ), 2)
    
def PSD(Af, df):
    """
        Power spectral density of a spectrum
    """
    return np.power( np.abs( Af ), 2) / df

def FFT( At, dt, axis = 1):   
    """
        Proper Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(At)))*dt

def IFFT( Af, dt,axis=1):
    """
        Proper Inverse Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Af)))/dt
    
def RFSpectrum( At, dt):
    """
        Radio Frequency spectrum (ie spectrum off of a photodiode). Note that we use the real FFT
    """
    return np.abs(np.fft.rfft(np.power( np.abs( At ), 2)))





def save_experiment(filename, experiment):
    """
        Save an Experiment class instance as a pkl file - which save all of the information of the experiment including setting and outputs
    """
    with open(filename+'.pkl', 'wb') as output:
        pickle.dump(experiment, output, pickle.HIGHEST_PROTOCOL)
#        pickle.dump(env, output, pickle.HIGHEST_PROTOCOL)
        
def load_experiment(filename):
    """
        Load an Experiment class instance from a pkl file 
    """
    with open(filename+'.pkl', 'rb') as input:
        experiment = pickle.load(input)
#        env = pickle.load(input)
    return experiment

            


def savelogbook(logbook, filepath):
    """
        Saves the GA logbook as a csv file for future reference
    """
    import pandas as pd
    df_log = pd.DataFrame(logbook)
    df_log.to_csv(filepath + '.csv', index=False)
    
    
def extractlogbook(logbook):
    """
        Loads a GA logbook from a csv file 
    """
    hrlog = list(logbook[0].keys())
    log = {}
   
    for stat in hrlog:
        log[stat] = [item[stat] for item in logbook] 

    return log



def splitindices(num, div):
    """
        Used when distributing the tasks in the GA to various cores. This function provides the indices which will evenly split a list of length 'num' into 'div' divisions
    """
    indices = [0]
    for i in range(div):
        val = num//(div - i)
        num += -val
        prev = indices[-1]
        indices.append(val + prev)
    return indices




def recurse(E, node, out_edges, path, startpoints, in_edges):
    """
        This is a recursive function which, based on certain rules of causality/etc finds the proper order to apply the components in, on a given graph. It is important that all preceeding components are simulated first. For example, before a splitter can/should be simulated, both paths leading into it must be dealt with. This function (called from Experiment.make_path()) recurses through the graph and produces a list of subpaths, which provides instructions on the order to simulate the components in.
    """
    path_i = []   
    
    # loop until conditions met (if for some reason there's an error, Python has fail-safes - but still be cafeful)
    while True:
        
        # add node (should happen only once)
        path_i.append(node)
        
        # if we are at a termination node (no successors), we can finish and check other, unfinished paths
        if len(E.suc(node)) == 0:
            break        
        
        # if we're not at a splitter, we can continue on the path no problem and add it to this sub_path
        if not E.nodes[E.suc(node)[0]]['info'].splitter:
            node = E.suc(node)[0]
        
        # if we ARE at a splitter, we need to start a new sub_path and recurse the function again
        else:
            break
    
    # we've traversed through one subpath, let's add it to our list of paths
    path.append(path_i)
    
    ## now we need to figure out where to start from for the next recursion
    
    # we're again at a termination node, look for unfinished paths
    if len(E.suc(node)) == 0:
        
        # if there's more unfinished paths, set one as our current node, remove it from the list and recurse
        if len(out_edges) > 0: 
            node = out_edges[0]; out_edges.pop(0)
            (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
            
        # if there's NO MORE unfinished paths, but we have unchecked startpoint nodes (input nodes), go to one of those, set it as the current node, and recurse again
        if len(startpoints) != 0:
            node = startpoints[0]; startpoints.pop(0)
            (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
    
    # if we're not at a termination point, append the current node to the current subpath
    else:
        node = E.suc(node)[0]
        
        # if this is the first time we encounter this node, we list it's output nodes and increment our count of how many times we've encountered this node
        if in_edges[node] == 0:
            for s in E.suc(node): out_edges.append(s)
        in_edges[node] += 1 
        
        # if we have now we've run into this node the right number of times, we add it to the subpath as we can now simulate it
        if in_edges[node] == len(E.pre(node)):
            path.append([node])
            
            # we go back and check if there are more unfinished paths. If there is, select one and set our node there, remove if from the list, and recurse
            if len(out_edges) > 0:
                node = out_edges[0]
                out_edges.pop(0)
                (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
            
        # check if we still haven't run through all the incoming paths to the current node, we loop back again
        elif in_edges[node] != len(E.pre(node)): ## finished loop?
            if len(out_edges) > 0:
                node = out_edges[0]
                out_edges.pop(0)
                (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
            
        # if we've made it this far, we've finished all the way through from one startpoint, and we choose another startpoint and recurse
        if len(startpoints) != 0:
            node = startpoints[0]
            startpoints.pop(0)
            (E, node, out_edges, path, startpoints, in_edges) = recurse(E, node, out_edges, path, startpoints, in_edges)
    
    # once we've reached this far, we have traversed the whole graph and built instructions on how to simulate the graph (experiment)
    return (E, node, out_edges, path, startpoints, in_edges)


