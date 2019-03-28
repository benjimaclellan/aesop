import pickle
import numpy as np
#from assets.classes import Experiment

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

def FFT( At, dt, ax = 0):   
    """
        Proper Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(At, axes=ax), axis=ax), axes=ax)*dt

def IFFT( Af, dt, ax=0):
    """
        Proper Inverse Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Af, axes=ax), axis=ax), axes=ax)/dt
    
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

            
def reload_experiment(filename):
    """
        Reloads an experiment from filename
    """
    E = rebuild_experiment( load_experiment(filename) )
    return E
    

def rebuild_experiment(E):
    adj = list(E.edges())
    components = {}
    measurement_nodes = []
    for node in E.nodes:
        components[node] = E.nodes[node]['info'].__class__()
        if len(E.suc(node)) == 0:
            measurement_nodes.append(node)
    experiment = Experiment()
    experiment.buildexperiment(components, adj, measurement_nodes)
    experiment.checkexperiment()
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

