import autograd.numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import random

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

# ---------------------------- Data Generation ---------------------------

def generate_data():
    graph = get_graph()
    propagator = get_propagator()
    evaluator = get_evaluator()

    # pure GA
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    ga_pop, ga_log = tuning_genetic_algorithm(graph, propagator, evaluator, n_population=5, n_generations=5)
    ga_log.to_pickle('GA_default_log.pkl')
    with open('GA_default_pop.pkl', 'wb') as handle:
        pickle.dump(ga_pop, handle)
    print("GA")
    print(ga_log)

    # pure Adam
    np.random.seed(RANDOM_SEED)
    adam_pop, adam_log = tuning_adam_gradient_descent(graph, propagator, evaluator, n_pop=5, n_batches=5, batch_size=10)
    adam_log.to_pickle('Adam_default_log.pkl')
    with open('Adam_default_pop.pkl', 'wb') as handle:
        pickle.dump(adam_pop, handle)
    print("Adam")
    print(adam_log)
    
    # use GA population to begin Adam tuning
    ga_adam_pop, ga_adam_log = tuning_adam_gradient_descent(graph, propagator, evaluator, n_pop=5, n_batches=5, batch_size=10, pop=ga_pop)
    ga_adam_log.to_pickle('GA_Adam_log.pkl')
    with open('GA_Adam_pop.pkl', 'wb') as handle:
        pickle.dump(ga_adam_pop, handle)
    print('GA + Adam')
    print(ga_adam_log)