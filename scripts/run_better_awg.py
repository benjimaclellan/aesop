"""
Test of topology optimization routines
"""

# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform
import copy
import numpy as np
from scipy.interpolate import interp1d

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# various imports
import matplotlib.pyplot as plt
import psutil

from lib.functions import InputOutput

from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evolver import ProbabilityLookupEvolver

from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink

from simulator.fiber.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from simulator.fiber.node_types_subclasses.inputs import ContinuousWaveLaser
from simulator.fiber.node_types_subclasses.outputs import Photodiode
from simulator.fiber.node_types_subclasses.single_path import PhaseModulator, IntensityModulator
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof

from lib.functions import parse_command_line_args

plt.close('all')
if __name__ == '__main__':
    propagator = Propagator(window_t=20e-9, n_samples=2**14, central_wl=1.55e-6)

    pattern_rep_rate = 1e9  # speed that the full pattern repeats
    num_bits = 10
    bit_rate = pattern_rep_rate * num_bits  # rate of the details, i.e. bit rate
    minimum_mod_rate = 2e9  # how slow can the modulators go. this should be more than the pattern rep rate
    amplitude = 0.02
    bit_depth = 2**4

    for mod in (IntensityModulator, PhaseModulator):
        mod.min_frequency = minimum_mod_rate
        mod.step_frequency = 1.0e9

    total_reps = np.ceil(propagator.window_t * pattern_rep_rate).astype('int')

    rs = np.random.RandomState(seed=0)
    bits = rs.randint(0, bit_depth, num_bits) / bit_depth * amplitude
    ybits = np.tile(bits, total_reps)
    xbits = np.arange(propagator.t[0], propagator.t[-1], 1/bit_rate)

    ypatt = interp1d(xbits, ybits, kind='nearest', fill_value="extrapolate")(propagator.t)

    fig, ax = plt.subplots(1,1)
    ax.scatter(xbits, ybits)
    ax.plot(propagator.t, ypatt, color='salmon')
    ax.set(xlim=[propagator.t[0], propagator.t[-1]], xlabel='Time')

    options_cl = parse_command_line_args(sys.argv[1:])

    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path='rfawg_better', unique_id=True)
    io.save_machine_metadata(io.save_path)

    ga_opts = {'n_generations': 14,
               'n_population': psutil.cpu_count()-1,
               'n_hof': 6,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}


    evaluator = RadioFrequencyWaveformGeneration(propagator, target_harmonic=pattern_rep_rate,
                                                 target_amplitude=None,
                                                 target_waveform=ypatt,)
    evolver = ProbabilityLookupEvolver(verbose=False)

    pd = Photodiode()
    pd.protected = True

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             'sink': TerminalSink()}
    edges = {('source', 0): ContinuousWaveLaser(),
             (0, 'sink'): pd,
             }
    graph = Graph.init_graph(nodes=nodes, edges=edges)

    graph.assert_number_of_edges()
    graph.initialize_func_grad_hess(propagator, evaluator)

    update_rule = 'roulette'
    hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                     ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                     parameter_opt_method='L-BFGS+GA',
                                     include_dashboard=False, crossover_maker=None)

    save_hof(hof, io)
    plot_hof(hof, propagator, evaluator, io)

    fig, ax = plt.subplots(1, 1, figsize=[5,3])
    ax.fill_between(log['generation'], log['best'], log['mean'], color='grey', alpha=0.2)
    ax.plot(log['generation'], log['best'], label='Best')
    ax.plot(log['generation'], log['mean'], label='Population mean')
    ax.plot(log['generation'], log['minimum'], color='darkgrey', label='Population minimum')
    ax.plot(log['generation'], log['maximum'], color='black', label='Population maximum')
    ax.set(xlabel='Generation', ylabel='Cost')
    ax.legend()

    io.save_fig(fig, 'topology_log.png')
