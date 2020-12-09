import pickle
import matplotlib.pyplot as plt
import random
import autograd.numpy as np

# necessary for successful imports below -------------------
import sys
import pathlib
import os
import platform
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)
# ----------------------------------------------------------
from problems.example.evolution_operators.evolution_operators import RemoveOneInterferometerPath

random.seed(0)
np.random.seed(5)

def get_saved_design(graph_pickle, evaluator_pickle, propagator_pickle):
    with open(graph_pickle, 'rb') as graph_handle:
        with open(evaluator_pickle, 'rb') as evaluator_handle:
            with open(propagator_pickle, 'rb') as propagator_handle:
                graph = pickle.load(graph_handle)
                evaluator = pickle.load(evaluator_handle)
                propagator = pickle.load(propagator_handle)

                return graph, propagator, evaluator


def display_saved_design(graph, propagator):
    graph.draw()
    plt.show()
    graph.propagate(propagator)
    graph.inspect_state(propagator)
    graph.display_noise_contributions(propagator)


def modify_saved_design():
    # this is super particular to the design I want to modify
    graph, propagator, evaluator = get_saved_design('graph_hof0.pkl', 'evaluator.pkl', 'propagator.pkl')
    graph.remove_edge(3, 6)
    graph.remove_edge(6, 7)
    graph.remove_edge(6, 4)
    graph.add_edge(3, 4)
    graph.remove_node(6)
    graph.remove_edge(4, 7)
    graph.remove_edge(7, 2)
    graph.remove_node(7)
    graph.add_edge(4, 2)

    # # graph = RemoveOneInterferometerPath().apply_evolution_at(graph, 4)
    # graph = RemoveOneInterferometerPath().apply_evolution(graph)
    graph.assert_number_of_edges()
    print(f'graph nodes: {graph.nodes}')
    print(f'graph edges: {graph.edges}')
    display_saved_design(graph, propagator)


if '__main__'==__name__:
    modify_saved_design()