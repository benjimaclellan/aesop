import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import pickle
import random
import pandas as pd
import time

from algorithms.parameter_optimization_utils import tuning_genetic_algorithm, tuning_adam_gradient_descent, get_individual_score, adam_bounded, get_initial_population, adam_function_wrapper
from lib.hessian import function_wrapper

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper, DelayLine

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import power_, psd_

from problems.example.assets.additive_noise import AdditiveNoise

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

"""
Note: picking the wrong index in phase shift (see evaluator_rfawg.py) can result in excessively discontinuous results in gradient descent
"""

# ---------------------------- Providers --------------------------------
def get_graph():
    """
    Returns the default graph for testing, with fixed topology at this time
    """
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1, 'central_wl': 1.55e-6, 'osnr_dB':55}),
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: WaveShaper(),
             3: DelayLine(),
             4: MeasurementDevice()
             }
    edges = [(0, 1, DispersiveFiber(parameters=[0])),
             (1, 2, DispersiveFiber(parameters=[0])),
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

def generate_data_GA_Adam_comparison(with_noise=True, resample_per_individual=False, resample_period_gen=1, resample_period_optimization=1, qualifier_str=''):    
    graph = get_graph()
    propagator = get_propagator()
    graph.propagate(propagator) # necessary atm to set up correct # of samples in noise
    evaluator = get_evaluator()

    # pure GA
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    ga_pop, ga_log = tuning_genetic_algorithm(graph, propagator, evaluator, noise=with_noise,
                                              resample_per_individual=resample_per_individual,
                                              resample_period_gen=resample_period_gen)
    ga_log.to_pickle(f'GA_default_log{qualifier_str}.pkl')
    with open(f'GA_default_pop{qualifier_str}.pkl', 'wb') as handle:
        pickle.dump(ga_pop, handle)
    print("GA")
    print(ga_log)

    # # pure Adam
    np.random.seed(RANDOM_SEED)
    adam_pop, adam_log = tuning_adam_gradient_descent(graph, propagator, evaluator, verbose=True, noise=with_noise,
                                                      resample_per_individual=resample_per_individual,
                                                      resample_period=resample_period_optimization)
    with open(f'Adam_default_log{qualifier_str}.pkl', 'wb') as handle:
        pickle.dump(adam_log, handle)
    with open(f'Adam_default_pop{qualifier_str}.pkl', 'wb') as handle:
        pickle.dump(adam_pop, handle)
    print("Adam")
    print(adam_log)
    
    # use GA population to begin Adam tuning
    with open(f'GA_default_pop{qualifier_str}.pkl', 'rb') as handle:
        ga_pop = pickle.load(handle)

        ga_adam_pop, ga_adam_log = tuning_adam_gradient_descent(graph, propagator, evaluator, noise=with_noise,
                                                                pop=ga_pop, verbose=True,
                                                                resample_per_individual=resample_per_individual,
                                                                resample_period=resample_period_optimization)
        ga_adam_log.to_pickle(f'GA_Adam_log{qualifier_str}.pkl')
        with open(f'GA_Adam_pop{qualifier_str}.pkl', 'wb') as handle2:
            pickle.dump(ga_adam_pop, handle2)
        print('GA + Adam')
        print(ga_adam_log)

    # Use Adam tuning within GA population for top 10 individuals of each generation
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    ga_with_adam_pop, ga_with_adam_log, ga_with_adam_adamLog = tuning_genetic_algorithm(graph, propagator, evaluator, optimize_top_X=10,
                                                                                        resample_per_individual=resample_per_individual,
                                                                                        resample_period_gen=resample_period_gen, noise=with_noise,
                                                                                        resample_period_optimization=resample_period_optimization)
    ga_with_adam_log.to_pickle(f'GA_with_Adam_log{qualifier_str}.pkl')
    with open (f'GA_with_Adam_pop{qualifier_str}.pkl', 'wb') as handle:
        pickle.dump(ga_with_adam_pop, handle)
    with open(f'GA_with_Adam_AdamLog{qualifier_str}.pkl', 'wb') as handle:
        pickle.dump(ga_with_adam_adamLog, handle)
    print('GA with Adam at each step')
    print(ga_with_adam_log)
    print(ga_with_adam_adamLog)


def load_and_output_data_GA_Adam_comparison(with_noise=True):
    AdditiveNoise.simulate_with_noise = with_noise

    graph = get_graph()
    propagator = get_propagator()
    evaluator = get_evaluator()

    # # GA
    # ga_log = pd.read_pickle('GA_default_log.pkl')
    # print('GA log:')
    # print(ga_log)

    # with open('GA_default_pop.pkl', 'rb') as handle:
    #     ga_pop = pickle.load(handle)
    #     _, best_params = ga_pop[0]
        
    #     _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
    #     best_score = get_individual_score(graph, propagator, evaluator, best_params, node_edge_index, parameter_index) # saved score was accurate with or without noise: we want to check sometimes how one solution with(out) noise might compare to one with(out)
        
    #     display_output_sawtooth(evaluator, graph, best_params, propagator, title=f'GA default population best: {best_score}')
    
    # # Adam
    # adam_log = pd.read_pickle('Adam_default_log.pkl')
    # print('Adam log: ')
    # print(adam_log)

    # with open('Adam_default_pop.pkl', 'rb') as handle:
    #     adam_pop = pickle.load(handle)
    #     _, best_params = adam_pop[0]
    #     _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
    #     best_score = get_individual_score(graph, propagator, evaluator, best_params, node_edge_index, parameter_index)
    #     display_output_sawtooth(evaluator, graph, best_params, propagator, title=f'Adam default population best: {best_score}')
    #     _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
    #     score = get_individual_score(graph, propagator, evaluator, best_params, node_edge_index, parameter_index)
    #     print(f'calculated score: {score}')
    #     assert score == best_score
    
    # GA + (and then) Adam
    ga_adam_log = pd.read_pickle('GA_Adam_log.pkl')
    print('GA + Adam log: ')
    print(ga_adam_log)

    with open('GA_Adam_pop.pkl', 'rb') as handle:
        ga_adam_pop = pickle.load(handle)
        _, best_params = ga_adam_pop[0]
        _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
        best_score = get_individual_score(graph, propagator, evaluator, best_params, node_edge_index, parameter_index)
        display_output_sawtooth(evaluator, graph, best_params, propagator, title=f'GA + Adam population best: {best_score}')
    
    # #GA with Adam at each step
    # ga_with_adam_log = pd.read_pickle('GA_with_Adam_log.pkl')
    # print('GA with Adam at each step log: ')
    # print(ga_with_adam_log)

    # with open('GA_with_Adam_Adamlog.pkl', 'rb') as handle:
    #     ga_with_adam_adamLog = pickle.load(handle)
    #     for i, adamLog in enumerate(ga_with_adam_adamLog):
    #         print(f'Generation {i + 1}')
    #         print(adamLog)

    # with open('GA_with_Adam_pop.pkl', 'rb') as handle:
    #     ga_with_adam_pop = pickle.load(handle)
    #     _, best_params = ga_with_adam_pop[0]
    #     _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
    #     best_score = get_individual_score(graph, propagator, evaluator, best_params, node_edge_index, parameter_index)
    #     display_output_sawtooth(evaluator, graph, best_params, propagator, title=f'GA with Adam at each step population best: {best_score}')

# ---------------------------- Noise data -------------------------------
def get_noise_setting_benchmark():
    """
    Gets data comparing the different methods with no noise, noise (no resampling),
    noise (generational resampling and batch resampling),
    noise (individual sampling).
    """
    # # no noise
    generate_data_GA_Adam_comparison(with_noise=False, qualifier_str='_noNoise')

    # with noise (no resampling)
    generate_data_GA_Adam_comparison(with_noise=True, resample_per_individual=False,
                                     resample_period_gen=-1, resample_period_optimization=-1,
                                     qualifier_str='_noiseNoResample')
    
    # with noise (generational and batch resampling)
    generate_data_GA_Adam_comparison(with_noise=True, resample_per_individual=False,
                                     resample_period_gen=1, resample_period_optimization=1,
                                     qualifier_str='_noiseGenerationalBatchResample')
    
    # with noise (individual sampling)
    generate_data_GA_Adam_comparison(with_noise=True, resample_per_individual=True, resample_period_gen=-1,
                                     resample_period_optimization=-1, qualifier_str='_noiseIndividualResample')


def _noise_treatment_benchmark_runtime():
    # load in all data and compare runtimes
    log_types = ['GA_default_log', 'Adam_default_log', 'GA_Adam_log', 'GA_with_Adam_log']

    labels = ['GA', 'Adam', 'GA followed by Adam', 'GA with Adam']
    no_noise = []
    noise_no_resample = []
    noise_gen_resample = []
    noise_ind_resample = []

    for log_name in log_types:
        no_noise_pd = pd.read_pickle(f'{log_name}_noNoise.pkl')
        no_noise.append(no_noise_pd['runtime'].sum() + 1) # + 1 is to counter the -1 runtime of generation 0

        noise_no_resample_pd = pd.read_pickle(f'{log_name}_noiseNoResample.pkl')
        noise_no_resample.append(noise_no_resample_pd['runtime'].sum() + 1)

        noise_gen_resample_pd = pd.read_pickle(f'{log_name}_noiseGenerationalBatchResample.pkl')
        noise_gen_resample.append(noise_gen_resample_pd['runtime'].sum() + 1)

        noise_ind_resample_pd = pd.read_pickle(f'{log_name}_noiseIndividualResample.pkl')
        noise_ind_resample.append(noise_ind_resample_pd['runtime'].sum() + 1)
    
    x = np.arange(len(labels)) * 2
    width = 0.6

    _, ax = plt.subplots()
    ax.bar(x - 3 * width / 4, no_noise, width / 2, label='no noise')
    ax.bar(x - width / 4, noise_no_resample, width / 2, label='noise no resample')
    ax.bar(x + width / 4, noise_gen_resample, width / 2, label='noise generation resample')
    ax.bar(x + 3 * width / 4, noise_ind_resample, width / 2, label='noise individual resample')

    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime by algorithm and noise treatment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()


def _noise_treatment_compare_params():
    log_types = ['GA_default_pop', 'Adam_default_pop', 'GA_Adam_pop', 'GA_with_Adam_pop']
    graph = get_graph()
    attributes = graph.extract_attributes_to_list_experimental(['parameter_names', 'upper_bounds'], get_location_indices=True, exclude_locked=True)
    labels = attributes['parameter_names']
    upper_bounds = attributes['upper_bounds']

    fig, ax = plt.subplots(4, 1)
    fig.tight_layout(pad=1)

    x = np.arange(len(labels)) * 2
    width = 0.6

    for i, log_name in enumerate(log_types):
        with open(f'{log_name}_noNoise.pkl', 'rb') as handle:
            pop = pickle.load(handle)
            param = np.array(pop[0][1]) / upper_bounds
            ax[i].bar(x - 3 * width / 4, param, width / 2, label='no noise')
        
        with open(f'{log_name}_noiseNoResample.pkl', 'rb') as handle:
            pop = pickle.load(handle)
            param = np.array(pop[0][1]) / upper_bounds
            ax[i].bar(x - 1 * width / 4, param, width / 2, label='noise no resample')
        
        with open(f'{log_name}_noiseGenerationalBatchResample.pkl', 'rb') as handle:
            pop = pickle.load(handle)
            param = np.array(pop[0][1]) / upper_bounds
            ax[i].bar(x + 1 * width / 4, param, width / 2, label='noise generation resample')
        
        with open(f'{log_name}_noiseIndividualResample.pkl', 'rb') as handle:
            pop = pickle.load(handle)
            param = np.array(pop[0][1]) / upper_bounds
            ax[i].bar(x + 3 * width / 4, param, width / 2, label='noise individual resample')
        
        ax[i].set_title(log_name)
        ax[i].tick_params(labelsize=7)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels)
    
    ax[2].set_ylabel('Param value (normalized against max allowed value)')
    ax[1].legend()

    plt.show()


def _assess_fitness_with_rand_noise(graph, propagator, evaluator, params):
    AdditiveNoise.simulate_with_noise = True
    _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()

    sum = 0
    for _ in range(50):
        graph.resample_all_noise()
        sum += get_individual_score(graph, propagator, evaluator, params, node_edge_index, parameter_index)

    return sum / 50


def _noise_treatment_fitness_benchmark():
    # load in all data and compare runtimes
    log_types = ['GA_default_pop', 'Adam_default_pop', 'GA_Adam_pop', 'GA_with_Adam_pop']

    labels = ['GA', 'Adam', 'GA followed by Adam', 'GA with Adam']
    no_noise = []
    noise_no_resample = []
    noise_gen_resample = []
    noise_ind_resample = []

    graph = get_graph()
    propagator = get_propagator()
    graph.propagate(propagator)
    evaluator = get_evaluator()

    np.random.seed(3050100)
    for log_name in log_types:
        with open(f'{log_name}_noNoise.pkl', 'rb') as handle:
            pop = pickle.load(handle)
            params = np.array(pop[0][1])
            no_noise.append(_assess_fitness_with_rand_noise(graph, propagator, evaluator, params))

        with open(f'{log_name}_noiseNoResample.pkl', 'rb') as handle:
            pop = pickle.load(handle)
            params = np.array(pop[0][1])
            noise_no_resample.append(_assess_fitness_with_rand_noise(graph, propagator, evaluator, params))
        
        with open(f'{log_name}_noiseGenerationalBatchResample.pkl', 'rb') as handle:
            pop = pickle.load(handle)
            params = np.array(pop[0][1])
            noise_gen_resample.append(_assess_fitness_with_rand_noise(graph, propagator, evaluator, params))

        with open(f'{log_name}_noiseIndividualResample.pkl', 'rb') as handle:
            pop = pickle.load(handle)
            params = np.array(pop[0][1])
            noise_ind_resample.append(_assess_fitness_with_rand_noise(graph, propagator, evaluator, params))
    
    x = np.arange(len(labels)) * 2
    width = 0.6

    _, ax = plt.subplots()
    ax.bar(x - 3 * width / 4, no_noise, width / 2, label='no noise')
    ax.bar(x - width / 4, noise_no_resample, width / 2, label='noise no resample')
    ax.bar(x + width / 4, noise_gen_resample, width / 2, label='noise generation resample')
    ax.bar(x + 3 * width / 4, noise_ind_resample, width / 2, label='noise individual resample')

    ax.set_ylabel('Fitness of best output')
    ax.set_title('Fitness by algorithm and noise treatment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()


def compare_noise_setting_benchmark():
    """
    The idea is to see how well the design would perform in a real system: thus we will run the top individuals of the
    population with noise (arbitrary noise sources)

    1. Compare runtimes from each methods (order of magnitude for now, since I ran them all on my laptop with a bunch of tabs open)
        Let's make a fun bar graph (grouped by convergence type, with each type of noise treatment in each group)
    2. Compare the actual output designs: are they similar or not at all?
        So make a bar graph... divide: each param gets a row (subdivided by the noise techniques). Each algorithm gets a separate plot
    3. Compare the final performances of the best element (averaged over X tests with diff noise. Remember to seed!!)
        Do the little bar graph thing again!
    4. TODO: compare hessian for stability?
    """
    _noise_treatment_benchmark_runtime()
    _noise_treatment_compare_params()
    _noise_treatment_fitness_benchmark()


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
    ax.scatter(x, y)
    plt.show()


def generate_adam_convergence_data():
    propagator = get_propagator()
    evaluator = get_evaluator()
    graph = get_graph()

    _generate_adam_convergence_data(graph, propagator, evaluator, title_qualifier='')


def display_adam_convergence_data():
    cm = plt.get_cmap('brg')     # create colour map

    _, ax = plt.subplots()
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
