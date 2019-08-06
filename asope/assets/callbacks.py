#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
import uuid
import time
import pickle

#%% import custom modules
from assets.functions import save_class
from assets.graph_manipulation import remake_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

#%%
def save_experiment_and_plot(experiment, env, At, filepath=None, filename=None):
    if filepath == None:
        filepath = 'results/'
    if filename == None:
#        filename = str(uuid.uuid4())
        filename = time.strftime("%Y%m%d-%H%M%S")
        
    save_class(filepath+filename+'_experiment', experiment)
    
    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    
    experiment.draw(node_label='both', ax=ax[0])
    
    ax[1].plot(env.t, P(At), label='Generated')
    ax[1].plot(env.t, env.target, label='Target')
    ax[1].set_xlim([0, (10/env.target_harmonic)])
    plt.legend()
    
    experiment.visualize( env, measurement_node=None, ax1=ax[2], ax2=ax[3])
    
    with open(filepath+filename+'_figure.pkl', 'wb') as output:
        pickle.dump(fig, output, pickle.HIGHEST_PROTOCOL)

    plt.savefig(filepath+filename+'_figure.png')