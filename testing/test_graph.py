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
from problems.example.evolver import Evolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types import SourceNode, SinkNode

from problems.example.evolver import StochMatrixEvolver, SizeAwareMatrixEvolver, ReinforcementMatrixEvolver

from algorithms.parameter_optimization import parameters_optimize
from algorithms.topology_optimization import topology_optimization

plt.close('all')

if __name__ == '__main__':
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)

    nodes = {'source':SourceNode(),
             0:VariablePowerSplitter(),
             'sink':SinkNode()}

    edges = {('source', 0):ContinuousWaveLaser(),
             (0,'sink'):PhaseModulator(),
             }

    # nodes = {0: ContinuousWaveLaser(),
    #          1: None,
    #          2: None,
    #          3: None,
    #          -1: Photodiode()}
    #
    # edges = {(0, 1): PhaseModulator(),
    #          (0, 2): WaveShaper(),
    #          (0, -1): WaveShaper(),
    #          (2, 3): WaveShaper(),
    #          (1, 3): WaveShaper(),
    #          (1, -1): WaveShaper(),
    #          (1, 2): WaveShaper(),
    #          (3, -1): WaveShaper(),
    #          }

    graph = Graph(nodes, edges)
    # graph.propagate(propagator)

    fig, ax = plt.subplots(1,1)
    graph.draw(ax=ax)
    # nx.draw(graph)