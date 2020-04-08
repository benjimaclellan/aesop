
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

from problems.example.node_types_children.inputs import PulsedLaser
from problems.example.node_types_children.outputs import MeasurementDevice
from problems.example.node_types_children.single_path import CorningFiber
from problems.example.node_types_children.multi_path import PhaseModulator



if __name__ == "__main__":

    propagator = Propagator(window_t = 100e-12, n_samples = 2**14)

    nodes = {0:PulsedLaser(),
             1:PhaseModulator(parameters_from_name={'mod_depth':1, 'mod_freq':10}),
             2:PhaseModulator(parameters=[1]),
             3:MeasurementDevice(parameters=[])}
    edges = [(0,1, CorningFiber(parameters=[1])),
             (1,2, CorningFiber(parameters=[1])),
             (2,3, CorningFiber(parameters=[1])),
             (0,3)]

    graph = Graph(nodes, edges, propagate_on_edges = True)

    plt.close('all')
    nx.draw(graph, labels = dict(zip(graph.nodes, graph.nodes)))

    #%%
    graph.propagate(propagator)
    graph.assert_number_of_edges()

    print(configuration.EVOLUTION_OPERATORS)
    print(configuration.NODE_TYPES)
    print(configuration.NODE_TYPES_ALL)