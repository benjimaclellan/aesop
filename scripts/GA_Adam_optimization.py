import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import warnings
import pickle
import random
import pandas as pd
import time

from algorithms.parameter_optimization_utils import tuning_genetic_algorithm, tuning_adam_gradient_descent, get_individual_score, adam_bounded, get_initial_population, adam_function_wrapper
from lib.analysis.hessian import function_wrapper

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import power_, psd_, fft_, ifft_

from algorithms.assets.functions import logbook_update, logbook_initialize

RANDOM_SEED = 58923
RANDOM_SEED_ADAM = 3901190
DISPLAY_GRAPHICS = True

# for Adam convergence tests
TEST_SIZE_ADAM = 32
NUM_DATAPOINTS_ADAM = 200
ITER_PER_DATAPOINT_ADAM = 10

# for diagnosing uphill
TEST_SIZE_UPHILL = 32
NUM_DATAPOINTS_UPHILL = 500
ITER_PER_DATAPOINT_UPHILL = 1

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

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


def get_propagator():
    return Propagator(window_t = 1e-8, n_samples = 2**14, central_wl=1.55e-6)


def get_evaluator():
    return RadioFrequencyWaveformGeneration(get_propagator())

# ---------------------------- Data Visualisation ------------------------

def display_output_sawtooth(evaluator, graph, graph_parameters, propagator, title="Sawtooth"):
    if DISPLAY_GRAPHICS:
        pnts_displayed = propagator.n_samples // 10 # just so visualisation isn't too crowded

        _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
        graph.distribute_parameters_from_list(graph_parameters, node_edge_index, parameter_index)
        graph.propagate(propagator)
        state = graph.measure_propagator([node for node in graph.nodes if not graph.out_edges(node)][0])
        actual_output = power_(state)

        _, ax = plt.subplots()
        ax.plot(propagator.t[0:pnts_displayed], evaluator.target[0:pnts_displayed], label='target', color='r', lw=1)
        ax.plot(propagator.t[0:pnts_displayed], actual_output[0:pnts_displayed], label='actual', color='b', lw=1)
        ax.legend()
        plt.title(title)
        plt.show()
    
def compare_params(graph, propagator, evaluator, params_list, label_list, title=''):
    if DISPLAY_GRAPHICS:
        pnts_displayed = propagator.n_samples // 10

        _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
        _, ax = plt.subplots(2, 1)
        # ax.plot(propagator.t[0:pnts_displayed], evaluator.target[0:pnts_displayed], label='target', lw=1)

        for params, label in zip(params_list, label_list):
            graph.distribute_parameters_from_list(params, node_edge_index, parameter_index)
            graph.propagate(propagator)
            state = graph.measure_propagator([node for node in graph.nodes if not graph.out_edges(node)][0])
            actual_output = power_(state)
            ax[0].plot(propagator.t[0:pnts_displayed], actual_output[0:pnts_displayed], label=f'{label} time', lw=1)
            ax[1].plot(propagator.f, psd_(state, propagator.dt, propagator.df), label=f'{label} freq', lw=1)
            print(label)
            print(psd_(state, propagator.dt, propagator.df))
        
        ax[0].legend()
        ax[1].legend()
        plt.title(title)
        plt.show()


# ---------------------------- Data Generation ---------------------------

def generate_data_GA_Adam_comparison():
    graph = get_graph()
    propagator = get_propagator()
    evaluator = get_evaluator()

    # # pure GA
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    ga_pop, ga_log = tuning_genetic_algorithm(graph, propagator, evaluator) # , n_population=20, n_generations=5)
    ga_log.to_pickle('GA_default_log.pkl')
    with open('GA_default_pop.pkl', 'wb') as handle:
        pickle.dump(ga_pop, handle)
    print("GA")
    print(ga_log)

    # # pure Adam
    np.random.seed(RANDOM_SEED)
    adam_pop, adam_log = tuning_adam_gradient_descent(graph, propagator, evaluator, verbose=True) # , n_pop=20, n_batches=5, batch_size=10)
    adam_log.to_pickle('Adam_default_log.pkl')
    with open('Adam_default_pop.pkl', 'wb') as handle:
        pickle.dump(adam_pop, handle)
    print("Adam")
    print(adam_log)
    
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
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    ga_with_adam_pop, ga_with_adam_log, ga_with_adam_adamLog = tuning_genetic_algorithm(graph, propagator, evaluator, optimize_top_X=10)
    ga_with_adam_log.to_pickle('GA_with_Adam_log.pkl')
    with open ('GA_with_Adam_pop.pkl', 'wb') as handle:
        pickle.dump(ga_with_adam_pop, handle)
    with open('GA_with_Adam_AdamLog.pkl', 'wb') as handle:
        pickle.dump(ga_with_adam_adamLog, handle)
    print('GA with Adam at each step')
    print(ga_with_adam_log)
    print(ga_with_adam_adamLog)


def load_and_output_data_GA_Adam_comparison():
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
        _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
        score = get_individual_score(graph, propagator, evaluator, best_params, node_edge_index, parameter_index)
        print(f'calculated score: {score}')
        assert score == best_score
    
    # GA + (and then) Adam
    ga_adam_log = pd.read_pickle('GA_Adam_log.pkl')
    print('GA + Adam log: ')
    print(ga_adam_log)

    with open('GA_Adam_pop.pkl', 'rb') as handle:
        ga_adam_pop = pickle.load(handle)
        best_score, best_params = ga_adam_pop[0]
        display_output_sawtooth(evaluator, graph, best_params, propagator, title=f'GA + Adam population best: {best_score}')
    
    #GA with Adam at each step
    ga_with_adam_log = pd.read_pickle('GA_with_Adam_log.pkl')
    print('GA with Adam at each step log: ')
    print(ga_with_adam_log)

    with open('GA_with_Adam_Adamlog.pkl', 'rb') as handle:
        ga_with_adam_adamLog = pickle.load(handle)
        for i, adamLog in enumerate(ga_with_adam_adamLog):
            print(f'Generation {i + 1}')
            print(adamLog)

    with open('GA_with_Adam_pop.pkl', 'rb') as handle:
        ga_with_adam_pop = pickle.load(handle)
        best_score, best_params = ga_with_adam_pop[0]
        display_output_sawtooth(evaluator, graph, best_params, propagator, title=f'GA with Adam at each step population best: {best_score}')

# ---------------------------- Adam Diagnosis ---------------------------

def _adam_convergence_from_start(graph, propagator, evaluator, params,
                                 num_datapoints=50, iter_per_datapoint=100, convergence_check_period=None):
    """
    Returns an array of the score after each iteration, and total runtime
    """
    _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()

    y = np.zeros(num_datapoints)
    
    y[0] = get_individual_score(graph, propagator, evaluator, params, node_edge_index, parameter_index)

    start_time = time.time()

    m, v = None, None
    
    # get bounds
    lower_bounds, upper_bounds = graph.get_parameter_bounds()
    
    # get gradient stuff
    fitness_funct = function_wrapper(graph, propagator, evaluator, exclude_locked=True)
    adam_fitness_funct = adam_function_wrapper(fitness_funct)
    fitness_grad = grad(adam_fitness_funct)

    for i in range(1, num_datapoints):
        print(f'   datapoint: {i}')
        params, termination_iter, m, v = adam_bounded(lower_bounds, upper_bounds, fitness_grad, params,
                                                      num_iters=iter_per_datapoint,
                                                      convergence_check_period=convergence_check_period, m=m, v=v, 
                                                      verbose=True)

        if (termination_iter != iter_per_datapoint):
            print(f"Terminated early on datapoint {i}, iteration: {termination_iter}")
            y[i:num_datapoints] = np.nan # we just don't want it to display anything
            break

        y[i] = get_individual_score(graph, propagator, evaluator, params, node_edge_index, parameter_index)
    
    return y, time.time() - start_time

def _generate_adam_convergence_data(graph, propagator, evaluator, title_qualifier=''):
    """
    Set arbitrary data, and we shall observe how varying num_datapoints and iter_per_datapoints affects the convergence rate
    """
    data = []
 
    # get population
    np.random.seed(RANDOM_SEED_ADAM) # need this to be consistent across runs to compare different performances
    pop, _, _ = get_initial_population(graph, propagator, evaluator, TEST_SIZE_ADAM, 'uniform')

    for i, (_, param) in enumerate(pop):
        print(f'population element: {i}')
        param_arr = np.array(param)
        y, runtime = _adam_convergence_from_start(graph, propagator, evaluator,
                                                  param_arr,
                                                  num_datapoints=NUM_DATAPOINTS_ADAM,
                                                  iter_per_datapoint=ITER_PER_DATAPOINT_ADAM)
        data.append((y, runtime))

    with open(f'{NUM_DATAPOINTS_ADAM}datapoints_{ITER_PER_DATAPOINT_ADAM}iterPerDatapoint_{title_qualifier}.pkl', 'wb') as handle:
        pickle.dump(data, handle)


def display_initial_pop(size=TEST_SIZE_ADAM, seed=RANDOM_SEED_ADAM):
    propagator = get_propagator()
    evaluator = get_evaluator()
    graph = get_graph()

    np.random.seed(seed)
    pop, _, _ = get_initial_population(graph, propagator, evaluator, size, 'uniform')
    _, ax = plt.subplots()
    x = np.zeros(size)
    y = np.array([score for (score, individual) in pop])
    print(f'x: {x.shape}')
    print(f'y: {y.shape}')
    ax.scatter(x, y)
    plt.show()


def generate_single_param(seed=RANDOM_SEED_ADAM):
    propagator = get_propagator()
    evaluator = get_evaluator()
    graph = get_graph()

    np.random.seed(seed)
    print('single parameter\n')
    pop, _, _ = get_initial_population(graph, propagator, evaluator, 1, 'uniform')
    print(f'param val: {pop[0][1]} \n\nscore: {pop[0][0]}')


def generate_adam_convergence_data():
    propagator = get_propagator()
    evaluator = get_evaluator()
    graph = get_graph()

    _generate_adam_convergence_data(graph, propagator, evaluator, title_qualifier='')


def display_adam_convergence_data():
    cm = plt.get_cmap('brg')     # create colour map

    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', [cm(1.*i/TEST_SIZE_ADAM) for i in range(TEST_SIZE_ADAM)])
    ax.set_xlim(right=NUM_DATAPOINTS_ADAM * ITER_PER_DATAPOINT_ADAM)
    x = np.arange(0, NUM_DATAPOINTS_ADAM * ITER_PER_DATAPOINT_ADAM, ITER_PER_DATAPOINT_ADAM)
    with open(f'{NUM_DATAPOINTS_ADAM}datapoints_{ITER_PER_DATAPOINT_ADAM}iterPerDatapoint_.pkl', 'rb') as handle:
        data = pickle.load(handle)
        for i, run in enumerate(data):
            ax.plot(x, run[0], label=f'test {i}')

        plt.title(f'Adam convergence: {NUM_DATAPOINTS_ADAM} datapoints, {ITER_PER_DATAPOINT_ADAM} iterations/datapoint, seed: {RANDOM_SEED_ADAM}')
        ax.legend()
        plt.show()


def diagnose_uphill_case():
    graph = get_graph()
    propagator = get_propagator()
    evaluator = get_evaluator()

    np.random.seed(RANDOM_SEED_ADAM) # need this to be consistent across runs to compare different performances
    pop, _, _ = get_initial_population(graph, propagator, evaluator, 24, 'uniform')
    score, param_list = pop[23]
    y, runtime = _adam_convergence_from_start(graph, propagator, evaluator,
                                              np.array(param_list),
                                              num_datapoints=NUM_DATAPOINTS_UPHILL,
                                              iter_per_datapoint=ITER_PER_DATAPOINT_UPHILL,
                                              convergence_check_period=None)

    run_data = (y, runtime)
    
    with open(f'{NUM_DATAPOINTS_UPHILL}datapoints_{ITER_PER_DATAPOINT_UPHILL}iterPerDatapoint_uphillCase.pkl', 'wb') as handle:
        pickle.dump(run_data, handle)

    fig, ax = plt.subplots()
    x = np.arange(0, NUM_DATAPOINTS_UPHILL * ITER_PER_DATAPOINT_UPHILL, ITER_PER_DATAPOINT_UPHILL)
    ax.plot(x, run_data[0])
    ax.legend()
    plt.title(f'Convergence of a single point, with phase shift in rfawg evaluator (truncated not rounded)')
    plt.show()

def display_uphill_case():
    # setup graphs
    fig, ax = plt.subplots()
    x = np.arange(0, NUM_DATAPOINTS_UPHILL * ITER_PER_DATAPOINT_UPHILL, ITER_PER_DATAPOINT_UPHILL)

    with open(f'{NUM_DATAPOINTS_UPHILL}datapoints_{ITER_PER_DATAPOINT_UPHILL}iterPerDatapoint_uphillCase.pkl', 'rb') as handle:
        y, _ = pickle.load(handle)
        ax.plot(x, y, label='')
    
    ax.legend()
    plt.title('Single starting point')
    plt.show()

def compare_big_jump():
    prejump_params = np.array(
                    [5.15130495, 0.92105773, 0.99999999, 0.79299178, 0.99999999, 0.80663515,
                     2.64814918, 5.37367294, 0.78283939, 5.31339156, 6.15854142, 0.46009278,
                     0.56741198, 0.76225641, 0.26890699, 0.41031763, 0.46803960, 0.41965900,
                     0.99999999]
                    )
    postjump_params = np.array(
                    [5.15050071, 0.92059505, 0.99999999, 0.79362395, 0.99999999, 0.80623950,
                     2.64781833, 5.37252838, 0.78398131, 5.31226456, 6.15866165, 0.45890617,
                     0.56864731, 0.76171969, 0.26982987, 0.41053914, 0.46860968, 0.42059382,
                     0.99999999]
                    )
    diff = postjump_params - prejump_params
    print(diff)

    graph = get_graph()
    propagator = get_propagator()
    evaluator = get_evaluator()
    _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()

    prejump_score = get_individual_score(graph, propagator, evaluator, prejump_params, node_edge_index=node_edge_index, parameter_index=parameter_index)
    postjump_score = get_individual_score(graph, propagator, evaluator, postjump_params, node_edge_index=node_edge_index, parameter_index=parameter_index)

    compare_params(graph, propagator, evaluator, [prejump_params, postjump_params], [f'prejump: {prejump_score}', f'postjump: {postjump_score}'], title='Comparing pre-jump and post-jump results')