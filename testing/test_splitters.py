# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

import matplotlib.pyplot as plt

from lib.functions import InputOutput

from lib.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import power_

from problems.example.assets.additive_noise import AdditiveNoise

from problems.example.node_types_subclasses.inputs import PulsedLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import DispersiveFiber
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter, FrequencySplitter
from problems.example.node_types import TerminalSource, TerminalSink

plt.close('all')

AdditiveNoise.noise_on = True

if __name__ == '__main__':
    propagator = Propagator(window_t=100e-9, n_samples=2**15, central_wl=1.55e-6)
    io = InputOutput(directory='splitter_tests', verbose=True)
    io.init_save_dir(sub_path='frequency_splitter', unique_id=False)
    # nodes = {'source': TerminalSource(),
    #          0: VariablePowerSplitter(),
    #          1: VariablePowerSplitter(),
    #          'sink': TerminalSink()}

    nodes = {'source': TerminalSource(),
             0: FrequencySplitter(),
             1: FrequencySplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0, 0): PulsedLaser(parameters_from_name={'train': True, 't_rep': 1e-9,  'pulse_width': 10e-12}),
             # (0, 1, 0): PhaseModulator(parameters=[1, 6e9, 0, 0]),
             (0, 1, 0): DispersiveFiber(parameters=[0]),
             (0, 1, 1): DispersiveFiber(parameters=[0]),
             (0, 1, 2): DispersiveFiber(parameters=[0]),
             (0, 1, 3): DispersiveFiber(parameters=[0]),
             (1, 'sink', 0):MeasurementDevice(),
             }

    graph = Graph.init_graph(nodes=nodes, edges=edges)

    for node in graph.nodes:
        if type(graph.nodes[node]['model']) in (VariablePowerSplitter, FrequencySplitter):
            graph.nodes[node]['model'].update_attributes(graph.get_in_degree(node), graph.get_out_degree(node))
            print(node, graph.get_in_degree(node), graph.get_out_degree(node))

    attributes = graph.extract_attributes_to_list_experimental(['parameters', 'upper_bounds'])
    # attributes['parameters'] = graph.sample_parameters_to_list()
    graph.distribute_parameters_from_list(attributes['parameters'], attributes['models'], attributes['parameter_index'])
    graph.propagate(propagator)

    #%%
    fig, ax = plt.subplots(1, 1)
    graph.draw(ax=ax)

    state = graph.measure_propagator('sink')
    fig, ax = plt.subplots(1, 1)
    ax.plot(propagator.t, power_(state))
    plt.show()

    # io.save_object(graph, 'frequency_splitters.pkl')
    # io.save_object(propagator, 'propagator.pkl')