
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import autograd.numpy as np
import matplotlib.animation as animation

from lib.graph import Graph

from problems.example.evolver import Evolver

from problems.example.assets.propagator import Propagator

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice

#%%

plt.close('all')
if __name__ == "__main__":

    propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
    evaluator = RadioFrequencyWaveformGeneration(propagator)
    evolver = Evolver()
    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6}),
             -1:MeasurementDevice()}
    edges = [(0, -1)]

    graph = Graph(nodes, edges, propagate_on_edges = False)
    graph.assert_number_of_edges()

    #%%
    plot = True
    animate = False
    if plot:
        fig1, ax1 = plt.subplots(2, 1)
        fig2, ax2 = plt.subplots(1, 2, figsize=[15,7])

    def animate_func(i, graph, evaluator, propagator, fig=fig2):
        N_TRIES = 0
        while N_TRIES < 20:
            # try:
            print(N_TRIES)
            graph_tmp, evo_op_choice = evolver.evolve_graph(graph, evaluator, propagator, verbose=True)
            print(evo_op_choice)
            graph.assert_number_of_edges()
            graph = graph_tmp
            break
            # except:
            #     N_TRIES += 1
            #     continue
        graph.assert_number_of_edges()

        # if plot:
        #     if i % 2 == 0:
        #         fig, ax = (fig1, ax1[0])
        #     else:
        #         fig, ax = (fig1, ax1[1])
        #
        #     plt.figure(fig1.number)
        #     ax.cla()
        #     graph.draw(ax=ax)
        #     plt.show()
        #     plt.pause(0.25)
        #     # plt.waitforbuttonpress()

        graph.sample_parameters(probability_dist='uniform', **{'triangle_width': 0.1})
        parameters, node_edge_index, parameter_index, *_ = graph.extract_parameters_to_list()
        graph.distribute_parameters_from_list(parameters, node_edge_index, parameter_index)

        graph.propagate(propagator, save_transforms=True)

        if plot:
            plt.figure(fig.number)
            state = graph.measure_propagator(-1)
            for ax in ax2: ax.cla()
            graph.draw(ax=ax2[0])
            ax2[1].plot(propagator.t, np.power(np.abs(state), 2))
            ax2[1].set_ylim([0,1])
            if not animate:
                plt.show()
                plt.waitforbuttonpress()
                # plt.pause(0.25)

    if animate:
        anim = animation.FuncAnimation(fig2, animate_func, frames=150, init_func=None, fargs=(graph, evaluator, propagator),
                                        interval=20, blit=False, repeat=True)
        anim.save(r"C:\Users\benjamin\Documents\Communication - Papers\thesis\figs\dr_strange.gif", writer='imagemagick', fps=30, dpi=100)

    elif plot and not animate:
        for i in range(100):
            animate_func(i, graph, evaluator, propagator, fig=fig2)