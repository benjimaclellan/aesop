
# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lib.functions import InputOutput

from lib.graph import Graph
from lib.minimal_save import extract_minimal_graph_info, build_from_minimal_graph_info

from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink

from simulator.fiber.evaluator_subclasses.evaluator_pulserep import PulseRepetition

from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.outputs import MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import PhaseModulator
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization

from lib.functions import parse_command_line_args


def flatten_df_entry(df_sub):
    if len(df_sub) == 0:
        raise RuntimeError('No matching entries')
    # if len(df_sub) > 1:
    #     raise RuntimeWarning('This unique id is at in multiple places. This is not generally an issue, but just so you know')
    return df_sub.to_numpy()[0]


plt.close('all')

if __name__ == '__main__':
    io = InputOutput(verbose=False)
    io.init_load_dir(sub_path='test_topology_and_hof_tracking/reduced_graphs')

    d = []
    for path in io.load_path.glob('*.json'):
        json_data = io.load_json(path.name)
        d_row = dict(filename=str(path.name), current_uuid=json_data['current_uuid'], parent_uuid=json_data['parent_uuid'])
        d.append(d_row)
    df = pd.DataFrame(d)
    print(df)

    #%%
    target = 'graph_gen4_ind5.json'
    target_uuid = flatten_df_entry(df['current_uuid'].loc[df['filename'] == target])

    tree_path = []
    next_branch_accessible = True
    i = 0
    while next_branch_accessible:
        i += 1
        try:
            tree_path.append(target_uuid)
            next_uuid = flatten_df_entry(df['parent_uuid'].loc[df['current_uuid'] == target_uuid])
            next_branch_accessible = True
            target_uuid = next_uuid
        except RuntimeError as e:
            next_branch_accessible = False
    # clean up the tree path, remove the last (which is dummy uuid) and reverse to start at the root
    tree_path.pop(-1)
    tree_path.reverse()

    fig, axs = plt.subplots(4, 4)
    axs = axs.flatten()
    scores = []
    for i, current_uuid in enumerate(tree_path):
        filename = flatten_df_entry(df['filename'].loc[df['current_uuid'] == current_uuid])
        print(filename)
        json_data = io.load_json(filename)
        graph = build_from_minimal_graph_info(json_data)

        graph.draw(ax=axs[i])
        scores.append(json_data['score'])

    fig, ax = plt.subplots(1, 1)
    ax.plot(scores)