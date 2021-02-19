"""
Test of topology optimization routines

TODO: HoF should be taken prior to speciation!!!!!
"""

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

# various imports
import matplotlib.pyplot as plt

from lib.graph import Graph

from problems.example.node_types_subclasses.terminals import TerminalSource, TerminalSink

from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof

plt.close('all')

def run_experiment(evaluator, propagator, io, evolver, ga_opts, input_laser, param_opt='L-BFGS+GA'):

    io.init_save_dir(sub_path='pulse_rep_rate', unique_id=True)
    io.save_machine_metadata(io.save_path)

    md = MeasurementDevice()
    md.protected = True

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             'sink': TerminalSink()}
    edges = {('source', 0): input_laser,
             (0, 'sink'): md,
             }
    graph = Graph.init_graph(nodes=nodes, edges=edges)

    update_rule = 'tournament'
    hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                     ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                     parameter_opt_method=param_opt,
                                     include_dashboard=False, crossover_maker=None)

    save_hof(hof, io)
    plot_hof(hof, propagator, evaluator, io)

    fig, ax = plt.subplots(1, 1, figsize=[5, 3])
    ax.fill_between(log['generation'], log['best'], log['mean'], color='grey', alpha=0.2)
    ax.plot(log['generation'], log['best'], label='Best')
    ax.plot(log['generation'], log['mean'], label='Population mean')
    ax.plot(log['generation'], log['minimum'], color='darkgrey', label='Population minimum')
    ax.plot(log['generation'], log['maximum'], color='black', label='Population maximum')
    ax.set(xlabel='Generation', ylabel='Cost')
    ax.legend()

    io.save_fig(fig, 'topology_log.png')