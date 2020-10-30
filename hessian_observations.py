import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
import random

from problems.example.assets.propagator import Propagator
from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration
from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser, PulsedLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, EDFA, CorningFiber, VariableOpticalAttenuator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.graph import Graph

from lib.hessian import get_scaled_hessian, plot_eigenvectors, lha_analysis
from algorithms.topology_optimization import parameters_optimize_complete

"""
Just to learn about things and look at effects

1. Try 2 setups that clearly could be reduced into one (i.e. 2 delay lines)
   Tests grouping components (lock together)
2. 2 setups that can less obviously be reduced (e.g. two waveshapers in a row)
   Tests same as 1
3. One setup where a certain component is doing nothing (e.g. beam splitter that's entirely biased to one side, the other branch is blank)
   Tests mechanism for potential path deletion
"""

np.random.seed(0)
random.seed(0)

propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
evaluator = RadioFrequencyWaveformGeneration(propagator)
# evaluator.evaluate_graph(graph, propagator)



def plot_hessian(H, info, title=''):
    fig, ax = plt.subplots()
    sns.heatmap(H)
    ax.set(xticks = [i+0.5 for i in range(len(info['parameters']))], yticks = [i+0.5 for i in range(len(info['parameters']))])
    ax.set_xticklabels(info['parameter_names'], rotation=45, ha='center', va='top')
    ax.set_yticklabels(info['parameter_names'], rotation=45, ha='right', va='center')
    plt.title(title)
    plt.show()

def run_setup(graph):
    _, graph = parameters_optimize_complete((None, graph), evaluator, propagator)
    graph.display_noise_contributions(propagator)
    params = graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True, exclude_locked=True)
    H_diag, H, H_evals, H_evecs = lha_analysis(graph.hess, params['parameters'])
    plot_hessian(H, params, title="Hessian")

    # print(f'Hdiag: \n{H_diag}')
    # print(f'H: \n{H}')
    # print(f'H eigenvals: \n{H_evals}')
    # print(f'H eigenvects: \n{H_evecs}')
    # plot_hessian(H_diag, params, title="diagonalized Hessian")


def archetypical_design():
    nodes = {0:ContinuousWaveLaser(),
             1: PhaseModulator(),
             2: WaveShaper(),
             -1: MeasurementDevice()
    }
    edges = [(0, 1), (1, 2), (2, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()
    run_setup(graph)


def setup_for_reduction():
    nodes = {0:ContinuousWaveLaser(),
             1: PhaseModulator(),
             2: CorningFiber(),
             3: CorningFiber(),
             4: WaveShaper(),
             -1: Photodiode()
    }
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, -1)]
    graph = Graph(nodes=nodes, edges=edges, propagate_on_edges=False)
    graph.assert_number_of_edges()

    run_setup(graph)

def numpy_help():
    arr = np.arange(0, 12, step=1).reshape(4, 3)
    print(f'arr: {arr}')
    print(f'arr[2][0]: {arr[2][0]}')
    print(f'arr[2, 0]: {arr[2, 0]}')
    print(f'arr[:][2]: {arr[:][2]}')
    print(f'arr[:, 2]: {arr[:, 2]}')
    print(f'arr[2][:]: {arr[2][:]}')
    print(f'arr[2,:]: {arr[2, :]}')



if __name__=='__main__':
    numpy_help()
    # archetypical_design()
    # setup_for_reduction()