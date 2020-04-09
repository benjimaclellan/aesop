
import matplotlib.pyplot as plt
plt.style.use(r"config/plot-style.mplstyle")

import networkx as nx
import itertools

import config.config as configuration
from config.config import np

from problems.example.node_types import Input
from problems.example.evaluator import Evaluator
from problems.example.graph import Graph
from problems.example.evolution_operators import EvolutionOperators
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_

from problems.example.node_types_children.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_children.outputs import MeasurementDevice
from problems.example.node_types_children.single_path import CorningFiber, PhaseModulator
from problems.example.node_types_children.multi_path import VariablePowerSplitter


if __name__ == "__main__":

    propagator = Propagator(window_t = 1e-7, n_samples = 2**14)

    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             1:PhaseModulator(parameters_from_name={'depth':1, 'frequency':2e9}),
             2:PhaseModulator(parameters_from_name={'depth':15, 'frequency':5e9}),
             3:MeasurementDevice()}
    edges = [(0,1, CorningFiber(parameters=[10])),
             (1,2, CorningFiber(parameters=[50])),
             (2,3, CorningFiber(parameters=[5]))]

    graph = Graph(nodes, edges, propagate_on_edges = True)

    plt.close('all')
    # nx.draw(graph, labels = dict(zip(graph.nodes, graph.nodes)))
    # plt.show()

    #%%
    graph.propagate(propagator)
    # graph.assert_number_of_edges()

    # print(configuration.EVOLUTION_OPERATORS)
    # print(configuration.NODE_TYPES)
    # print(configuration.NODE_TYPES_ALL)

#%%

    fig, ax = plt.subplots(2, 1)
    for node in reversed(graph.propagation_order):
        state = graph.nodes[node]['states'][0]
        ax[0].plot(propagator.t, power_(state), label=node)
        ax[1].plot(propagator.f, psd_(state, propagator.dt, propagator.df))

    ax[0].legend()
    plt.show()
