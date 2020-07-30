
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
import matplotlib.animation as animation

import config.config as configuration

from problems.example.evaluator import Evaluator
from problems.example.graph import Graph

from problems.example.evolver import Evolver

from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

#%%

plt.close('all')
if __name__ == "__main__":


    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    evolver = Evolver()
    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             -1:MeasurementDevice()}
    edges = [(0,-1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()

    #%%
    plot = True
    simulate = True
    animate = False
    if plot:
        fig1, ax1 = plt.subplots(2, 1)
        if simulate:
            fig2, ax2 = plt.subplots(1, 2, figsize=[15,7])

    def animate(i, graph, evaluator, propagator):
        while True:
            try:
                graph = evolver.evolve_graph(graph, evaluator, propagator)
                graph.assert_number_of_edges()
                break
            except:
                continue
        print('\n propagation order {}'.format(graph.propagation_order))
        # model_names = ['{}||{}'.format(graph.nodes[node]['model'].__class__.__name__, node) for node in graph.nodes]
        model_names = ['{}'.format(node) for node in graph.nodes]

        # if plot:
        #     if i % 2 == 0:
        #         fig, ax = (fig1, ax1[0])
        #     else:
        #         fig, ax = (fig1, ax1[1])
        #
        #     plt.figure(fig1.number)
        #     ax.cla()
        #     graph.draw(ax=ax, labels=dict(zip(graph.nodes, model_names)))
        #     plt.show()
        #     plt.pause(0.25)
        #     # plt.waitforbuttonpress()



        if simulate:
            graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
            parameters, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
            graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)

            graph.propagate(propagator, save_transforms=True)

            if plot:
                plt.figure(fig2.number)
                state = graph.measure_propagator(-1)
                for ax in ax2: ax.cla()
                model_names = ['{}'.format(node) for node in graph.nodes]
                graph.draw(ax=ax2[0], labels=dict(zip(graph.nodes, model_names)))
                ax2[1].plot(propagator.t, np.power(np.abs(state), 2))

    if animate:
        anim = animation.FuncAnimation(fig2, animate, frames=150, init_func=None, fargs=(graph, evaluator, propagator),
                                        interval=20, blit=False, repeat=True)
        anim.save(r"C:\Users\benjamin\Documents\Communication - Papers\thesis\figs\dr_strange.gif", writer='imagemagick', fps=30, dpi=100)













# # %%
# graph = 0
# nodes = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None, 6:None}
# # edges = [(0,1),
# #          (1,2),
# #          (2,5), (1,3), (3,4), (4,5), (5,6)]
# edges = [(0,1),
#          (1,5),
#          (1,5), (5,2), (2,3), (3,4), (4,6)]
# graph = Graph(nodes, edges, propagate_on_edges=False)
#
# source_node = 1
# sink_node = 5
# branch_to_keep = 3
#
# paths_from_source_to_sink = list(nx.all_simple_paths(graph, source_node, sink_node))
#
# paths_to_remove = [path for path in paths_from_source_to_sink if branch_to_keep not in path]
# nodes_to_remove = set([item for sublist in paths_to_remove for item in sublist])
# paths_to_keep = [path for path in paths_from_source_to_sink if branch_to_keep in path]
# nodes_to_keep = set([item for sublist in paths_to_keep for item in sublist]) - set([source_node, sink_node])
#
# if not nodes_to_keep:
#     graph.add_edge(graph.pre(source_node)[0], graph.suc(sink_node)[0])
# else:
#     graph.add_edge(graph.pre(source_node)[0], list(nodes_to_keep)[0])
#     graph.add_edge(list(nodes_to_keep)[-1], graph.suc(sink_node)[0])
#
# graph.remove_nodes_from(nodes_to_remove)
# print(graph.edges)
# plt.close('all')
# plt.figure()
# nx.draw_networkx(graph, with_labels=True)
# print(paths_from_source_to_sink)