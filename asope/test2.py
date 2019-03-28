import copy 
from random import shuffle, sample
import time
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import networkx as nx
import keyboard

from assets.functions import extractlogbook, save_experiment, load_experiment, splitindices, reload_experiment
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from assets.classes import Experiment
plt.close("all")

env = PulseEnvironment(p = 2, q = 1, profile = 'gauss')

components = {
    0:PowerSplitter(),
    3:PowerSplitter(),
    1:PhaseModulator(),
    2:Fiber(),
}
adj = [(0, 3), (0, 1), (1, 2), (2, 3)]
measurement_nodes=[3]


## now let's create the experiment as a custom class, build it based on specifications, and ensure it is properly connected
E = Experiment()
E.buildexperiment(components, adj, measurement_nodes)
E.checkexperiment()

E.make_path()
E.check_path()

for node in E.nodes():
    if not E.pre(node):
        E.nodes[node]['input'] = env.At
        
at = E.newattributes()
E.setattributes(at)

E.simulate(env)

E.measure(env, measurement_nodes[0], check_power=True)
plt.show()
