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
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from simulator.fiber.assets.additive_noise import AdditiveNoise

from simulator.fiber.node_types_subclasses.inputs import ContinuousWaveLaser
from simulator.fiber.node_types_subclasses.outputs import MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, IntensityModulator
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter, FrequencySplitter
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink

# from problems.example.evolver import ProbabilityLookupEvolver, SizeAwareLookupEvolver, ReinforcementLookupEvolver

plt.close('all')

AdditiveNoise.noise_on = False
ContinuousWaveLaser.protected = True

if __name__ == '__main__':
    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9,
                                                 target_amplitude=0.02, target_waveform='saw',)
    io = InputOutput(directory='interactive', verbose=True)
    io.init_save_dir(sub_path='example', unique_id=False)

    nodes = {'source': TerminalSource(),
             0: FrequencySplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             'sink': TerminalSink()}

    edges = {('source', 0): ContinuousWaveLaser(parameters=[1]),
             (0, 1, 0): PhaseModulator(parameters=[1, 6e9, 0, 0]),
             (0, 1, 1): DispersiveFiber(),
             # (0, 1, 2): DispersiveFiber(),
             # (1, 2, 0): DispersiveFiber(),
             (1, 2, 1): DispersiveFiber(),
             (1, 2, 2): IntensityModulator(),
             (2, 'sink'):MeasurementDevice(),
             }

    graph = Graph.init_graph(nodes=nodes, edges=edges)
    graph.update_graph()

    test = graph.extract_attributes_to_list_experimental(attributes=['parameters'])

    attributes = graph.extract_attributes_to_list_experimental(['parameters', 'upper_bounds'])
    parameters = graph.sample_parameters_to_list()
    graph.distribute_parameters_from_list(parameters, attributes['models'], attributes['parameter_index'])
    graph.propagate(propagator)
    graph.initialize_func_grad_hess(propagator, evaluator)
    # f = graph.func(parameters)
    # g = graph.grad(parameters)
    # h = graph.hess(parameters)
    # fig, ax = plt.subplots(1,1)
    # graph.draw(ax=ax, debug=True)
    #
    # state = graph.measure_propagator('sink')
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(propagator.t, power_(state))
    # plt.show()

    io.save_object(graph, 'test_graph1.pkl')
    io.save_object(propagator, 'propagator.pkl')
    io.save_object(evaluator, 'evaluator.pkl')