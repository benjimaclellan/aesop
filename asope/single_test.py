import time
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import networkx as nx

from assets.functions import extractlogbook, plot_individual, save_experiment, load_experiment, splitindices
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from assets.classes import Experiment
plt.close("all")

"""
A test script for simulating a single experiment, with either user-input parameters or randomly generated. Reference the README file for details of how to use the features.
"""

# this class stores all the information about the pulse
env = PulseEnvironment(p = 3, q = 2)

# define components wanted in the setup
components = (
    {
     'fib_input': Fiber(),
     'freq_split':FrequencySplitter(),
     'left':Fiber(),
     'right':Fiber(),
     'coupler':PowerSplitter(),
    }
    )

# specify how the components are connected together (first index -> second index)
adj = [('fib_input','freq_split'), ('freq_split','left'), ('freq_split','right'), ('left','coupler'), ('right','coupler')]

# which nodes do you want to measure at?
measurement_nodes = ['coupler']

# what parameters do you want on the components? Or you can randomly generate parameters later
attributes = {
                'fib_input':[0],
                'freq_split':[0],
                'left':[10],
                'right':[0]
             }


# now let's create the experiment as a custom class, build it based on specifications, and ensure it is properly connected
E = Experiment()
E.buildexperiment(components, adj, measurement_nodes)
E.checkexperiment()

# plot the graph structure of the experiment to check
E.draw(titles = 'both')

# calculate how to traverse the graph properly
E.make_path()
E.check_path()
E.print_path()

# if you want random parameters, uncomment the next line
#attributes = E.newattributes()

# dial-in the parameters (attributes) to each component
E.setattributes(attributes)

# here we initialize what pulse we inject into each starting node
for node in E.nodes():
    if len(E.pre(node)) == 0:
        E.nodes[node]['input'] = env.At

# here's where all the hard work happens and the transformations are simulated
E.simulate(env)

# plot the output of each measurement node
for measurement_node in E.measurement_nodes:
    E.measure(env, measurement_node)    
plt.show()

# here, we could save the experiment for further use if wanted
if False:
    save_experiment('test_experiment', E)
