import autograd.numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import random
import pandas as pd

from algorithms.parameter_optimization_utils import tuning_genetic_algorithm, tuning_adam_gradient_descent

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import power_

from algorithms.assets.functions import logbook_update, logbook_initialize

RANDOM_SEED = 58923
DISPLAY_GRAPHICS = True

# ---------------------------- Providers --------------------------------
def get_graph():
    """
    Returns the default graph for testing, with fixed topology at this time
    """
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1, 'central_wl': 1.55e-6}),
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: WaveShaper(),
             3: DelayLine(),
             4: MeasurementDevice()
             }
    edges = [(0,1, CorningFiber(parameters=[0])),
             (1,2, CorningFiber(parameters=[0])),
             (2,3),
             (3,4)]

    graph = Graph(nodes, edges, propagate_on_edges = True)
    graph.assert_number_of_edges()
    return graph


def get_propagator():
    return Propagator(window_t = 1e-8, n_samples = 2**14, central_wl=1.55e-6)


def get_evaluator():
    return RadioFrequencyWaveformGeneration(get_propagator())

# ---------------------------- Data Visualisation ------------------------

def display_output_sawtooth(evaluator, graph, graph_parameters, propagator, title="Sawtooth"):
    if DISPLAY_GRAPHICS:
        pnts_displayed = propagator.t.shape[0] // 10 # just so visualisation isn't too crowded

        _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
        graph.distribute_parameters_from_list(graph_parameters, node_edge_index, parameter_index)
        graph.propagate(propagator)
        state = graph.nodes[len(graph.nodes) - 1]['states'][0]
        actual_output = power_(state)
        
        # actual_normalized = (actual_output - np.min(actual_output)) / (np.max(actual_output) - np.min(actual_output))
        actual_normalized = actual_output / np.max(actual_output)
        expected_normalised = evaluator.target / np.max(evaluator.target)

        _, ax = plt.subplots()
        ax.plot(propagator.t[0:pnts_displayed], expected_normalised[0:pnts_displayed], label='target', color='r', lw=1)
        ax.plot(propagator.t[0:pnts_displayed], actual_normalized[0:pnts_displayed], label='actual', color='b', lw=1)
        ax.legend()
        plt.title(title)
        plt.show()

# ---------------------------- Data Generation ---------------------------

def generate_data():
    graph = get_graph()
    propagator = get_propagator()
    evaluator = get_evaluator()

    # # pure GA
    # np.random.seed(RANDOM_SEED)
    # random.seed(RANDOM_SEED)
    # ga_pop, ga_log = tuning_genetic_algorithm(graph, propagator, evaluator) # , n_population=20, n_generations=5)
    # ga_log.to_pickle('GA_default_log.pkl')
    # with open('GA_default_pop.pkl', 'wb') as handle:
    #     pickle.dump(ga_pop, handle)
    # print("GA")
    # print(ga_log)

    # # pure Adam
    # np.random.seed(RANDOM_SEED)
    # adam_pop, adam_log = tuning_adam_gradient_descent(graph, propagator, evaluator, verbose=True) # , n_pop=20, n_batches=5, batch_size=10)
    # adam_log.to_pickle('Adam_default_log.pkl')
    # with open('Adam_default_pop.pkl', 'wb') as handle:
    #     pickle.dump(adam_pop, handle)
    # print("Adam")
    # print(adam_log)
    
    # use GA population to begin Adam tuning
    with open('GA_default_pop.pkl', 'rb') as handle:
        ga_pop = pickle.load(handle)

        ga_adam_pop, ga_adam_log = tuning_adam_gradient_descent(graph, propagator, evaluator, pop=ga_pop, verbose=True) #, n_pop=20, n_batches=5, batch_size=10, pop=ga_pop)
        ga_adam_log.to_pickle('GA_Adam_log.pkl')
        with open('GA_Adam_pop.pkl', 'wb') as handle2:
            pickle.dump(ga_adam_pop, handle2)
        print('GA + Adam')
        print(ga_adam_log)

    # # Use Adam tuning within GA population for top 10 individuals of each generation
    # np.random.seed(RANDOM_SEED)
    # random.seed(RANDOM_SEED)
    # ga_from_adam_pop, ga_from_adam_log = tuning_genetic_algorithm(graph, propagator, evaluator, optimize_top_X=10)
    # ga_from_adam_log.to_pickle('GA_with_Adam_log.pkl')
    # with open ('GA_with_Adam_pop.pkl', 'wb') as handle:
    #     pickle.dump(ga_from_adam_pop, handle)
    # print('GA with Adam')
    # print(ga_with_adam_log)


def load_and_output_data():
    graph = get_graph()
    propagator = get_propagator()
    evaluator = get_evaluator()

    # GA
    ga_log = pd.read_pickle('GA_default_log.pkl')
    print('GA log:')
    print(ga_log)

    with open('GA_default_pop.pkl', 'rb') as handle:
        ga_pop = pickle.load(handle)
        best_score, best_params = ga_pop[0]
        display_output_sawtooth(evaluator, graph, best_params, propagator, title=f'GA default population best: {best_score}')
    
    # Adam
    adam_log = pd.read_pickle('Adam_default_log.pkl')
    print('Adam log: ')
    print(adam_log)

    with open('Adam_default_pop.pkl', 'rb') as handle:
        adam_pop = pickle.load(handle)
        best_score, best_params = adam_pop[0]
        display_output_sawtooth(evaluator, graph, best_params, propagator, title=f'Adam default population best: {best_score}')
    
    # GA + (and then) Adam
    ga_adam_log = pd.read_pickle('GA_Adam_log.pkl')
    print('GA + Adam log: ')
    print(ga_adam_log)

    with open('GA_Adam_pop.pkl', 'rb') as handle:
        ga_adam_pop = pickle.load(handle)
        best_score, best_params = ga_adam_pop[0]
        display_output_sawtooth(evaluator, graph, best_params, propagator, title=f'GA + Adam population best: {best_score}')
