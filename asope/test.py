

#%%
import copy 
from random import shuffle, sample
import time
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import networkx as nx
import keyboard
import peakutils
import matplotlib
from scipy.signal import find_peaks

from assets.functions import extractlogbook, save_experiment, load_experiment, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, AmplitudeModulator
from assets.classes import Experiment

from assets.environment import shift_function

from assets.graph_manipulation import *

plt.close("all")

import warnings
warnings.filterwarnings("ignore")

#%%
for i in range(0,1):
    plt.close('all')
    
    env = PulseEnvironment()
    
    components = {
    0:PhaseModulator(),
    1:WaveShaper(),
    }
    adj = [(0,1)]
    measurement_nodes=[1]
    
#    components = {
#    0:PhaseModulator(),
#    }
#    adj = []
#    measurement_nodes=[0]
    
    E = Experiment()
    E.buildexperiment(components, adj, measurement_nodes)
    
#    E = Experiment()
#    E,_ = brand_new_experiment(E)
    
#    for i in range(5):
#        E, _ = change_experiment_wrapper(E)
    
    E.cleanexperiment()
    E.checkexperiment()
    
    E.make_path()
    E.check_path()
    
#    E.draw(node_label='both')
    
    for node in E.nodes():
        if not E.pre(node):
            E.nodes[node]['input'] = env.At
    
    E.measurement_nodes = []
    for node in E.nodes():
        if not E.suc(node):
            E.measurement_nodes.append(node)
        
#    at = E.newattributes()

#    ### square
#    profile = 'square'
#    at = {0: [0.9721562922995752], 1: [0.9388051718035044, 0.6088379431538495, 0.44300448042663376, 0.8322380738846585, 0.7586872615024143, 0.8099102597981979, 0.8596685244158648, 0.22272785074548296, 0.5216859597298762, 0.9755623599891335, 0.938593424675552, 0.28125417790689966, 0.4628255676566311, 1.5486061906758934, 1.018568622326115, 0.9363228392479723, 1.2667142560333635, 1.0234001630377982, 1.2518182427028366, 1.604328002947039, 0.9328053293760543, 1.0505301936836304, 0.3848030732042502, 1.3465977750629976, 1.4238962556495751, 1.3807153739459812]}


    ## high-speed sine
#    profile = 'sine'
#    at = {0: [1.4437262344722013], 1: [0.9619170712171516, 0.22853899144845669, 0.862685822411654, 1.1841148199595974, 1.0107294553204766, 0.8839715205165168]}
    
    # sawtooth
    profile = 'saw'
    at = {0: [0.921105542125884], 1: [0.7126379453153524, 0.4709107806477839, 0.8074519670580008, 0.4806786400786607, 0.7240310500469911, 0.9141944067899248, 0.7758523041369754, 0.3737296838951669, 0.28090663453951736, 0.529949469630066, 0.7660490943030537, 0.5794509798963499, 0.24467908351966605, 0.5598712254564906, 0.8356774733152903, 0.2540840056898912, 1.1996192326062054, 1.4582472778805768, 0.21962025906698535, 0.24450941163934203, 0.8480224037619122, 1.29218129782412, 0.19667162635202518, 0.413165124310662, 1.150405959742855, 1.9567523580539008]}
    
    
    E.setattributes(at)
    E.simulate(env)
    E.visualize(env, E.measurement_nodes[0])
    plt.show()
    
    At = E.nodes[E.measurement_nodes[0]]['output']

    E.measure(env, E.measurement_nodes[0], check_power=True)
#    fit = env.fitness(At)
    
#    plt.show()
#    plt.pause(0.05)
    
    PAt = P(At)
    psd = PSD(FFT(At, env.dt), env.df)

    #%%
    ## plot freq parameters
    colors = ['salmon', 'teal', 'orange', 'green']
    lwidth = 2
    filepath = "/home/benjamin/Documents/Work - Courses/INRS/New Horizons in Photonics/Project/aiultrafast_manuscript/figures/"
    
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)
    
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 14), dpi=60)
    plt.subplots_adjust(hspace=0.6,wspace=0)
    ax[0].plot(env.t/1e-9, P(env.At0), label='Input Waveform', color=colors[3], lw=lwidth, alpha=0.6)
    ax[0].plot(env.t/1e-9, env.target, label='Target Waveform', color=colors[1], lw=lwidth)
    ax[0].plot(env.t/1e-9, PAt, label='Generated Waveform', color=colors[0], lw=lwidth)
    
    if profile == 'sine':
        ax[0].set_xlim([0, 0.20])
    else:
        ax[0].set_xlim([0, 0.50])
        
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Amplitude (arb. units)')
    ax[0].legend(loc=1)
    
    ax[1].plot(env.t/1e-9, E.nodes[0]['info'].lines[0][1], label='Phase Modulation Pattern', color=colors[0], lw=lwidth)
    if profile == 'sine':
        ax[1].set_xlim([0, 0.20])
    else:
        ax[1].set_xlim([0, 0.50])
        
    ax[1].set_xlabel('Time (ns)')
    ax[1].set_ylabel('Modulation Depth (rad)')
    ax[1].legend(loc=1)
    
    ax2 = ax[2].twinx()
    lns1 = ax[2].plot(env.f/1e9, E.nodes[1]['info'].lines[0][1], label='Amplitude Mask', color=colors[1])
    lns2 = ax2.plot(env.f/1e9, E.nodes[1]['info'].lines[1][1], label='Phase Mask', color=colors[2])
    ax[2].plot(env.f/1e9, psd/max(psd), lw=lwidth, color=colors[3], alpha=0.5, ls='--')
    ax[2].set_xlabel('Frequency Shift (GHz)')
    ax[2].set_ylabel('Amplitude')
    ax2.set_ylabel('Phase Shift (rad)')
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax[2].legend(lns, labs)
    ax[2].set_xlim([-250,250])

    import string
    for n,axs in enumerate(ax):
        axs.text(0.02, 1.12,'{})'.format(string.ascii_uppercase[n]),
     horizontalalignment='center',
     verticalalignment='center',
     transform = axs.transAxes,
     size=26, weight='bold')
    
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    plt.show()
    
#    plt.savefig(filepath+'result_{}.png'.format(profile),bbox='tight')
     
    #%%
    shifted = shift_function(env.target, env.target_rfft, PAt, env.array, env.target_harmonic, env.target_harmonic_ind, env.dt)
    
#    plt.figure()
#    plt.plot(env.t,env.target)
#    plt.plot(env.t,PAt)
#    plt.plot(env.t, shifted)
#    
#    print(fit)
#    peakinds = peakutils.indexes(rf)
#    peakf_dist = np.mean( self.df * np.diff(peakinds) )
    
#%%
"""
env = PulseEnvironment(p = 2, q = 1, profile = 'cw')

E = Experiment()
E,_ = brand_new_experiment(E)

fig1 = plt.figure()
fig2 = plt.figure()

for i in range(60):
    for j in range(2):
        E, _ = change_experiment_wrapper(E)
    E.cleanexperiment()
    E.make_path()
    E.check_path()
    
    mapping=dict(zip( [item for sublist in E.path for item in sublist],range(0,len(E.nodes()))))
    E = nx.relabel_nodes(E, mapping) 
    
    E.make_path()
    E.check_path()
    E.checkexperiment()   

    plt.figure(fig1.number)
    plt.clf()
    E.draw(node_label='disp_name', title='Optical Setup Mutation')#, fig=fig1)
    
#    plt.savefig('results/{}_{}_graph.png'.format(i, env.profile), bbox='tight', dpi=300)
    plt.show()
    
    
    E.injection_nodes = []
    E.measurement_nodes = []
    for node in E.nodes():
        if len(E.pre(node)) == 0:
            E.injection_nodes.append(node)
            E.nodes[node]['input'] = env.At
        if len(E.suc(node)) == 0:
            E.measurement_nodes.append(node)
        
    E.make_path()
    print(E.path)
    E.check_path()
        
    
    
    at = E.newattributes()
    E.setattributes(at)
    
    plt.figure(fig2.number)
    E.simulate(env)
    power_check = E.measure(env, E.measurement_nodes[0], fig=fig2, check_power = True)  
    plt.show()
#    plt.savefig('results/{}_{}_output.png'.format(i, env.profile), bbox='tight', dpi=300)

    if not power_check:
        raise ValueError

    adj = list(E.edges())
    components = {}
    measurement_nodes = []
    for node in E.nodes:
#        components[node] = eval(E.nodes[node]['info'].__class__.__name__)()
        components[node] = E.nodes[node]['info'].__class__.__name__
        if len(E.suc(node)) == 0:
            measurement_nodes.append(node)
    print('components = {')
    for node in components:
        print('{}:{}(),'.format(node, components[node]))
    print('}}\nadj = {}\nmeasurement_nodes={}'.format(adj, measurement_nodes))


    plt.pause(0.002)
"""
