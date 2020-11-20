# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform
import copy

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

import networkx as nx
import itertools
import os
from pathlib import Path
import random
import time
import string
from datetime import date
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np

import config.config as configuration

from problems.example.evaluator import Evaluator
from problems.example.evolver import ProbabilityLookupEvolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.assets.additive_noise import AdditiveNoise

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types import TerminalSource, TerminalSink

from problems.example.evolver import ProbabilityLookupEvolver, SizeAwareLookupEvolver, ReinforcementLookupEvolver

from algorithms.parameter_optimization import parameters_optimize
from algorithms.topology_optimization import topology_optimization

plt.close('all')


AdditiveNoise.noise_on = False
ContinuousWaveLaser.protected = True

if __name__ == '__main__':
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)

    # nodes = {'source':TerminalSource(),
    #          0:VariablePowerSplitter(),
    #          'sink':TerminalSink()}
    #
    # edges = {('source', 0):ContinuousWaveLaser(),
    #          (0,'sink'):Photodiode(),
    #          }

    # num_inputs, num_outputs = 6, 10
    # states = VariablePowerSplitter().propagate(num_inputs*[propagator.state], propagator, num_inputs, num_outputs)
    # print(len(states))
    # print(states[0].shape)

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): ContinuousWaveLaser(parameters=[1]),
             (0, 1, 0): PhaseModulator(parameters=[1, 6e9, 0, 0]),
             (0, 1, 1): DispersiveFiber(parameters=[0]),
             (1, 'sink'):MeasurementDevice(),
             }

    graph = Graph.init_graph(nodes=nodes, edges=edges)
    test = graph.extract_attributes_to_list_experimental(attributes=['parameters'])

    attributes = graph.extract_attributes_to_list_experimental(['parameters', 'upper_bounds'])
    parameters = graph.sample_parameters_to_list()
    graph.distribute_parameters_from_list(parameters, attributes['models'], attributes['parameter_index'])
    graph.propagate(propagator)

    fig, ax = plt.subplots(1,1)
    graph.draw(ax=ax, debug=True)

    state = graph.measure_propagator('sink')
    fig, ax = plt.subplots(1, 1)
    ax.plot(propagator.t, power_(state))
    plt.show()