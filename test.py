
import unittest
import copy
import psutil
import ray
import autograd.numpy as np
import autograd
import warnings
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import random

import config.config as config

from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.evolver import ProbabilityLookupEvolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_
from problems.example.assets.additive_noise import AdditiveNoise

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter
from problems.example.node_types import TerminalSource, TerminalSink

from problems.example.evolution_operators.evolution_operators import AddSeriesComponent, RemoveComponent, SwapComponent, AddParallelComponent

class Test(unittest.TestCase):
    def test_all_available_nodes(self):
        propagator = Propagator(window_t=100e-12, n_samples=2 ** 14)
        for node_type, nodes in config.NODE_TYPES_ALL.items():
            print('Testing {} node-types'.format(node_type))
            for node_name, node in nodes.items():
                tmp = node().propagate([propagator.state], propagator)
                print('\tTesting {} node'.format(node_name))
            print('\n')
        return

    def test_evaluation_operators(self):
        n_repeats, n_evolutions = 2, 2
        propagator = Propagator(window_t=1e-9, n_samples=2 ** 14, central_wl=1.55e-6)
        evaluator = RadioFrequencyWaveformGeneration(propagator)
        evolver = Evolver()
        nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1, 'central_wl': 1.55e-6}),
                 -1: MeasurementDevice()}
        edges = [(0, -1)]

        base_graph = Graph(nodes, edges, propagate_on_edges=False)
        base_graph.assert_number_of_edges()
        base_graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

        for i in range(n_repeats):
            graph = copy.deepcopy(base_graph)
            for j in range(n_evolutions):
                # print(f"repeat {i}, evolution number {j}")
                graph, evo_op_choice = evolver.evolve_graph(graph, evaluator, propagator)
                graph.assert_number_of_edges()
        return


def test_differentiability():
    check_warnings = True
    if check_warnings:
        warnings.filterwarnings('error')
        np.seterr(all='raise')

    propagator = Propagator(window_t=1e-9, n_samples=2 ** 14, central_wl=1.55e-6)
    for node_class, node_sub_classes in config.NODE_TYPES_ALL.items():
        for node_sub_class, model_class in node_sub_classes.items():
            # print(f"Testing model {model_class} in subclass {node_sub_class}")
            def test_model_propagate(parameters, parameter_inds, model, propagator):
                for parameter, parameter_ind in zip(parameters, parameter_inds):
                    model.parameters[parameter_ind] = parameter
                model.set_parameters_as_attr()
                
                propagator.state = np.ones(propagator.n_samples).reshape(propagator.n_samples, 1) * 0.0001
                states = model.propagate([propagator.state], propagator)

                return np.abs(states[0])

            def mask_parameters_by_lock(attribute_list, parameter_locks):
                masked_attribute_list = [attribute for (attribute, locked) in zip(attribute_list, parameter_locks)
                                         if not locked]
                return masked_attribute_list


            model = model_class()

            function = lambda parameters: test_model_propagate(parameters, parameter_inds, model, propagator)
            gradient = autograd.elementwise_grad(function)

            default_parameters = mask_parameters_by_lock(model.parameters, model.parameter_locks)
            upper_bounds = mask_parameters_by_lock(model.upper_bounds, model.parameter_locks)
            lower_bounds = mask_parameters_by_lock(model.lower_bounds, model.parameter_locks)
            parameter_inds = mask_parameters_by_lock(list(range(model.number_of_parameters)), model.parameter_locks)

            for parameters, parameters_name in zip([default_parameters, upper_bounds, lower_bounds], ["default_parameters", "upper_bounds", "lower_bounds"]):
                try:
                    function(parameters)
                    gradient(parameters)
                except (Exception, RuntimeWarning) as error:
                    print(f"differentiability errors on model {model_class}, parameter set of {parameters_name}\n\t{error}\n")
    return


def test_differentiability_graphical(include_locked=True):
    propagator = Propagator(window_t=1e-9, n_samples=2 ** 14, central_wl=1.55e-6)
    results = []

    amp_func = lambda output : np.mean(np.abs(output)) + np.std(np.abs(output))
    phase_func = lambda output: np.mean(np.angle(output)) + np.std(np.angle(output))
    power_func = lambda output: np.mean(power_(output)) + np.std(power_(output))
    ifft_fft_func = lambda output: np.mean(np.abs(np.fft.ifft(np.fft.fft(output))))
    test_function_list = [amp_func, phase_func, power_func, ifft_fft_func]
    test_function_names = ['np.abs', 'np.angle', 'power_', 'ifft * fft']

    for _, node_sub_classes in config.NODE_TYPES_ALL.items():
        for _, model_class in node_sub_classes.items():
            model = model_class()
            for i in range(0, model.number_of_parameters):
                model = model_class()
                if not include_locked and model.parameter_locks[i]:
                    continue
                if type(model.parameters[i]) == float and model.lower_bounds[i] is not None and model.upper_bounds[i] is not None:
                    test_func_results = [_get_gradient_vectors(model, i, propagator, test_func, graphical_test='none', noise=True) for test_func in test_function_list]                    
                    record = {'model':model_class.__name__, 'param':model.parameter_names[i],'differentiable':test_func_results}
                    results.append(record)
    
    print('\n\nRESULTS (showing failures only)\n')
    for result in results:
        if False in result['differentiable']:
            failed_functions = [test_function_names[i] for i in range(len(test_function_list)) if not result['differentiable'][i]] 
            print(f"model {result['model']}, param {result['param']} differentiation failed for functions {failed_functions}")


def _get_gradient_vectors(model, param_index, propagator, eval_func, noise=True, steps=200, rtol=5e-2, atol=1e-9, graphical_test='if failing'):
    """
    Plot gradient vectors. One is computed with autograd, the other with np.diff (finite difference). Also plot the function

    If graphical_test == 'always', displays all plots
    If graphical_test == 'if failing', displays all plots where the gradient and finite difference had a significant difference
    If graphical_test == 'none', does not display plots
    """
    print(f'model: {model.node_acronym}, param: {model.parameter_names[param_index]}')
    AdditiveNoise.simulate_with_noise = noise
    if (model.lower_bounds[param_index] == model.upper_bounds[param_index]):
        return

    def lock_unwanted_params(node_model, unlock_index):
        for i in range(node_model.number_of_parameters):
            if i != unlock_index:
                node_model.parameter_locks[i] = True
            else:
                node_model.parameter_locks[i] = False
    
    def test_function(param):
        model.parameters[param_index] = param
        model.set_parameters_as_attr()
        model.update_noise_model()

        input_state = (np.ones(propagator.n_samples).reshape(propagator.n_samples, 1) + np.sin(2 * np.pi / propagator.window_t * propagator.t) +
                       np.sin(4 * np.pi / propagator.window_t * propagator.t) + np.cos(32 * np.pi / propagator.window_t * propagator.t)) * 0.005 
        output = model.propagate([input_state], propagator)[0]
        if (model.noise_model is not None):
            output = model.noise_model.add_noise_to_propagation(output, propagator)

        return eval_func(output) 

    gradient_func = autograd.grad(test_function)

    lock_unwanted_params(model, param_index)
    model.parameters = model.default_parameters

    param_vals, delta = np.linspace(model.lower_bounds[param_index], model.upper_bounds[param_index], num=steps, retstep=True)
    
    function_output = np.array([test_function(param_val) for param_val in param_vals])
    finite_diff_gradient_output = np.diff(function_output) / delta
    normalization_factor = delta / steps # normalize the x axis over a unit interval, for automated tests

    if hasattr(finite_diff_gradient_output, '_value'): # if it's an arraybox object, fix it
        finite_diff_gradient_output = finite_diff_gradient_output._value
    
    gradient_output = np.array([gradient_func(param_val) for param_val in param_vals])
    gradient_correct = _gradients_match(gradient_output * normalization_factor, finite_diff_gradient_output * normalization_factor, rtol=rtol, atol=atol, noise=noise and model.noise_model is not None)

    if graphical_test == 'always' or (graphical_test == 'if failing' and not gradient_correct):
        fig, ax = plt.subplots(2, 1)
        ax[1].plot(param_vals, gradient_output, label='autograd derivative')
        ax[1].plot(param_vals[0:-1] + delta / 2, finite_diff_gradient_output, label='finite difference derivative', ls='--')
        ax[0].plot(param_vals, function_output, label='function', color='black')
        ax[0].legend()
        ax[0].set_ylabel('f(x)')
        ax[1].set_ylabel('df/dx')
        ax[1].legend()
        plt.title(f'Autograd and FDM derivatives. Model {model.node_acronym}, param {model.parameter_names[param_index]}, range {model.lower_bounds[param_index]}-{model.upper_bounds[param_index]}')
        plt.draw()
        plt.pause(0.001)
        response = input('Evaluate plot. Plot correct [y/n]: ').lower()
        while (response != 'y' and response != 'n'):
            response = input(f'{response} is an invalid response. Plot correct [y/n]: ').lower()
        plt.close(fig)
        if response == 'y':
            return True
        else:
            return False
    
    return gradient_correct


def _gradients_match(gradient_output, finite_diff_output, rtol=5e-2, atol=1e-9, max_mean_dev=1e-5, max_std_dev=1e-5, noise=True):
    # the gradient is averaged, because the finite differences are closest to the derivative halfway between the parameter values
    # the first two points are ignored, because there's often slightly strange behaviour at 0
    direct_match = np.allclose((gradient_output[1:-1] + gradient_output[2:]) / 2, finite_diff_output[1:], rtol=rtol, atol=atol)
    if not noise or direct_match:
        return direct_match
    else:
        residuals = (gradient_output[1:-1] + gradient_output[2:]) / 2 - finite_diff_output[1:]
        _, p = shapiro(residuals)

        return np.abs(np.mean(residuals)) < max_mean_dev and np.std(residuals) < max_std_dev and shapiro(residuals)[1] > 0.05 # shapiro tests for normal distribution, which noise SHOULD be

    




# old code for checking differentiation warnings
    # print(f"***************************This graph passed with no runtime warnings or issues!")
    # print(f"graph models are:")
    # for node in graph.nodes: print(f"\t{graph.nodes[node]['model'].__class__}")
    # except (RuntimeWarning) as e:
    #     print(f"Getting a RuntimeError in the parameter_optimize_multiprocess\n\t{e}")
    #     print(f"graph models are:")
    #     for node in graph.nodes: print(f"\t{graph.nodes[node]['model'].__class__}")
    #     score = 6666


def test_distributed_computing():
    @ray.remote
    def ray_test_function(x, const):
        return x * x * const
    ray.init(num_cpus=psutil.cpu_count(), include_dashboard=False, ignore_reinit_error=True)
    const = 2
    const_id = ray.put(const)
    results = ray.get([ray_test_function.remote(x, const_id) for x in range(2*psutil.cpu_count())])
    ray.shutdown()
    return

def get_test_graph0():
    nodes = {'source':TerminalSource(),
            0:VariablePowerSplitter(),
            1:VariablePowerSplitter(),
            'sink':TerminalSink()}
    edges = {('source', 0):ContinuousWaveLaser(),
             (0, 1):PhaseModulator(),
             (1,'sink'):Photodiode(),
            }
    graph = Graph(nodes, edges)
    # graph.assert_number_of_edges()
    # print(f'graph edges: {graph.edges}')
    # print(f'graph interfaces: {graph.interfaces}')
    return graph

def test_evo_op(graph, evo_op):
    print(f'Allowed evolution locations for this comp:')
    possible_locations = evo_op.possible_evo_locations(graph)
    print(f'possible locations: {possible_locations}')
    location = random.choice(possible_locations)
    print(f'location: {location}')
    new_graph = evo_op.apply_evolution(graph, location)
    print(f'new graph: \n{new_graph}')
    for edge in graph.edges:
        print(f"edge: {edge}, model: {graph.edges[edge]['model'].__class__.__name__}")


def test_evo_op_add_comp_series():
    graph = get_test_graph0()
    print('ADD IN SERIES:')
    evo_op = AddSeriesComponent(verbose=True)
    test_evo_op(graph, evo_op)


def test_evo_op_remove_comp():
    graph = get_test_graph0()
    print('REMOVE COMP:')
    evo_op = RemoveComponent(verbose=True)
    test_evo_op(graph, evo_op)


def test_evo_op_swap_comp():
    graph = get_test_graph0()
    print('SWAP COMP:')
    evo_op = SwapComponent(verbose=True)
    test_evo_op(graph, evo_op)

def test_evo_op_add_comp_parallel():
    """
    TODO: test cases
    1. No parallel possible because not enough nodes
    2. No parallel possible due to max input/output size
    3. Multiple options!
    """
    graph = get_test_graph0()
    print(f'ADD IN PARALLEL:')
    evo_op = AddParallelComponent(verbose=True)
    test_evo_op(graph, evo_op)

def test_evolver_base_lookup():
    graph = get_test_graph0()
    evaluator = Evaluator()
    evolver = ProbabilityLookupEvolver(verbose=True)
    evolver.random_graph(graph, evaluator, view_evo=True, n_evolutions=20)

if __name__ == "__main__":
    random.seed(3)
    np.random.seed(5)
    test_evolver_base_lookup()
    # test_evo_op_add_comp_series()
    # test_evo_op_remove_comp()
    # test_evo_op_swap_comp()
    # test_evo_op_add_comp_parallel()
    # unittest.main()
    # test_differentiability()
    # test_differentiability_graphical(include_locked=True)
    # test_distributed_computing()