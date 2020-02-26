import pickle
import autograd.numpy as np
import matplotlib.pyplot as plt
"""
    A number of useful functions that are utilized through the package.
"""

def P(field):
    """
        Power of a signal, as the square of the absolute value
    """
    return np.power( np.abs( field ), 2)
    
def PSD(field, dt, df, ax = 0):
    """
        Power spectral density of a spectrum
    """
    return np.power( np.abs( FFT(field,dt,ax) ), 2) / df

def FFT( field, dt, ax = 0):   
    """
        Proper Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(field, axes=ax), axis=ax), axes=ax)*dt

def IFFT( Af, dt, ax=0):
    """
        Proper Inverse Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Af, axes=ax), axis=ax), axes=ax)/dt
    
def RFSpectrum( field, dt, ax=0):
    """
        Radio Frequency spectrum (ie spectrum off of a photodiode). Note that we use the real FFT
    """
    return np.abs(np.fft.rfft(np.power( np.abs( field ), 2), axis=ax))


def LowPassRF( field, env, cutoff, dt, ax=0):
    #field_f = FFT(P(field), dt)
    filter = 1.0/np.sqrt(1+np.power(np.abs(env.f)/cutoff, 12))
    field = IFFT(FFT(P(field), dt) * filter, env.dt)

    return field

def save_class(filename, experiment):
    """
        Save an Experiment class instance as a pkl file - which save all of the information of the experiment including setting and outputs
    """
    with open(filename+'.pkl', 'wb') as output:
        pickle.dump(experiment, output, pickle.HIGHEST_PROTOCOL)
#        pickle.dump(env, output, pickle.HIGHEST_PROTOCOL)
        
def load_class(filename):
    """
        Load an Experiment class instance from a pkl file 
    """
    with open(filename+'.pkl', 'rb') as input:
        experiment = pickle.load(input)
#        env = pickle.load(input)
    return experiment

            
def reload_experiment(experiment):
    """
        Reloads an experiment from filename
    """
    exp = rebuild_experiment( experiment )
    return exp
    

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


def hessian(x):
    x_prime = np.gradient(x)
    hess = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grd_k in enumerate(x_prime):
        # Take gradient of each component
        tmp = np.gradient(grd_k)
        for l, grd_kl in enumerate(tmp):
            hess[k, l, :, :] = grd_kl

    return hess
