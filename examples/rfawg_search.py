
import sys
sys.path.append('..')

import networkx as nx
import itertools
import os
import random
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
import pandas as pd

import config.config as config
from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.evolver import Evolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.parameter_builtin import parameters_optimize
from algorithms.topology_random_search import topology_random_search
# from algorithms.parameters_genetic_algorithm import parameters_genetic_algorithm

plt.close('all')
if __name__ == '__main__':
    io = InputOutput(directory='testing_topology', verbose=True)
    io.save_machine_metadata()

    # propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    # evaluator = RadioFrequencyWaveformGeneration(propagator)
    # evolver = Evolver()
    # nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
    #          1:PhaseModulator(),
    #          2:WaveShaper(),
    #          -1:MeasurementDevice()}
    # edges = [(0,1),(1,2),(2,-1)]
    #
    # graph = Graph(nodes, edges, propagate_on_edges = False)
    # graph.assert_number_of_edges()
    # graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
    #
    # graph, score, log = topology_random_search(graph, propagator, evaluator, evolver, io, multiprocess=True)

    # print(graph.get_graph_info())
    # if False:
    # io.init_save_dir(sub_path=None, unique_id=False)
    # io.save_graph(graph, 'test_graph.pkl')
    #
    # io.save_machine_metadata(sub_path=None)
    #
    # io.init_load_dir(sub_path=None)
    # graph_load = io.load_graph('test_graph.pkl')
    #
