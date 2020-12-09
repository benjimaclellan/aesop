import pickle 

from problems.example.graph import Graph
from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper, DelayLine, IntensityModulator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types import TerminalSource, TerminalSink
from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.assets.propagator import Propagator

def start_graph():
    nodes = {'source':TerminalSource(),
              0:VariablePowerSplitter(),
             'sink':TerminalSink()
            }
    edges = {('source', 0):ContinuousWaveLaser(),
             (0,'sink'):Photodiode()
            }
    graph = Graph.init_graph(nodes=nodes, edges=edges)
    graph.assert_number_of_edges()
    return graph


def IM_graph():
    nodes = {'source':TerminalSource(),
              0:VariablePowerSplitter(),
              1:VariablePowerSplitter(),
             'sink':TerminalSink()
            }
    edges = {('source', 0):ContinuousWaveLaser(),
             (0, 1): IntensityModulator(),
             (1,'sink'):Photodiode()
            }
    graph = Graph.init_graph(nodes=nodes, edges=edges)
    graph.assert_number_of_edges()
    return graph


def PM_WS_graph():
    nodes = {'source':TerminalSource(),
              0:VariablePowerSplitter(),
              1:VariablePowerSplitter(),
              2:VariablePowerSplitter(),
             'sink':TerminalSink()
            }
    edges = {('source', 0):ContinuousWaveLaser(),
             (0, 1): PhaseModulator(),
             (1, 2): WaveShaper(),
             (2,'sink'):Photodiode()
            }
    graph = Graph.init_graph(nodes=nodes, edges=edges)
    graph.assert_number_of_edges()
    return graph

# def god_tier_graph():
#     with open(f'god_tier_graph.pkl', 'rb') as handle:
#         god_tier = pickle.load(handle)
#         print(god_tier)


def make_pkl(graph, name=''):
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(graph, handle)

def make_evaluator_pickle():
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    make_pkl(evaluator, 'evaluator')

if __name__ == "__main__":
    make_evaluator_pickle()
    pass
    # god_tier_graph()
    # make_pkl(start_graph(), 'start_graph')
    # make_pkl(IM_graph(), 'IM_graph')
    # make_pkl(PM_WS_graph(), 'PM_WS_graph')