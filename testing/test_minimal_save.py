
import sys
sys.path.append('..')

import matplotlib.pyplot as plt

from lib.graph import Graph
from problems.example.assets.propagator import Propagator
from lib.functions import InputOutput

from lib.minimal_save import extract_minimal_graph_info, build_from_minimal_graph_info

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, OpticalAmplifier
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink

plt.close('all')
if __name__ == "__main__":
    io = InputOutput()
    io.init_save_dir(sub_path='test_minimal_saving', unique_id=False)
    io.init_load_dir(sub_path='test_minimal_saving')

    propagator = Propagator(window_t=10/12e9, n_samples=2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9, target_waveform='saw')

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             3: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): ContinuousWaveLaser(),
             (0, 1): PhaseModulator(),
             (1, 2): WaveShaper(),
             (2, 3): OpticalAmplifier(),
             (3, 'sink'): Photodiode(),
             }

    graph = Graph.init_graph(nodes, edges)
    graph.update_graph()
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    #%%
    method = 'L-BFGS+GA'
    graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
    print(graph.extract_parameters_to_list())

    #%%
    filename = r"graph_data.json"
    json_data = extract_minimal_graph_info(graph)
    json_data['current_uuid'] = 'current_graph_uuid'
    json_data['parent_uuid'] = 'parent_graph_uuid'
    io.save_json(json_data, filename)

    json_data = io.load_json(filename)
    graph = build_from_minimal_graph_info(json_data)
    print(graph.extract_parameters_to_list())
