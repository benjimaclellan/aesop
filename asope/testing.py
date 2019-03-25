import os

import time
import matplotlib.pyplot as plt
import multiprocess as mp
import uuid
import copy

from assets.functions import extractlogbook, save_experiment, load_experiment
from assets.environment import PulseEnvironment
from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
from assets.classes import Experiment, GeneticAlgorithmParameters

from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import finetune_individual

plt.close("all")

E = load_experiment('test4')



adj = list(E.edges())
components = {}
measurement_nodes = []
for node in E.nodes:
    components[node] = E.nodes[node]['info'].__class__()
    if len(E.suc(node)) == 0:
        measurement_nodes.append(node)
E = Experiment()
E.buildexperiment(components, adj, measurement_nodes)
E.checkexperiment()

E.cleanexperiment()

env = PulseEnvironment(p = 1, q = 2, profile = 'cw')


#components = (
#    {
#     'fiber0': Fiber(),
#     'ps0':PowerSplitter(),
#     'pm0':PhaseModulator(),
#     'fiber1':Fiber(),
#     'ps1':PowerSplitter(),
#     'fiber2':PowerSplitter(),
#    }
#    ) 
#adj = [('fiber0','ps0'), ('ps0','pm0'), ('pm0','ps1'), ('ps0','fiber1'), ('fiber1','ps1'), ('ps1', 'fiber2')]
#measurement_nodes = ['fiber2']
#
#E = Experiment()
#E.buildexperiment(components, adj, measurement_nodes)
#E.checkexperiment()
#



E.make_path()
print(E.path)
E.check_path()


E.draw(node_label='both')
plt.show()



E.injection_nodes = []
E.measurement_nodes = []
for node in E.nodes():
    if len(E.pre(node)) == 0:
        E.injection_nodes.append(node)
        E.nodes[node]['input'] = env.At
    if len(E.suc(node)) == 0:
        E.measurement_nodes.append(node)

at = E.newattributes()
E.setattributes(at)

E.simulate(env)
E.measure(env, E.measurement_nodes[0])

