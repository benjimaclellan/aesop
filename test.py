
import unittest
import copy
import psutil
import ray
import autograd.numpy as np
import autograd
import warnings
import matplotlib.pyplot as plt

import config.config as config

from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.evolver import Evolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_
from problems.example.assets.additive_noise import AdditiveNoise

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter


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


def test_differentiability_all_params():
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
            
            def lock_unwanted_params(model, attribute_list, unlock_index):
                for i in range(len(attribute_list)):
                    if i != unlock_index:
                        model.parameter_locks[i] = True
                    else:
                        model.parameter_locks[i] = False
                
                return [attribute_list[unlock_index]]

            
            model = model_class()

            function = lambda parameters: test_model_propagate(parameters, parameter_inds, model, propagator)
            gradient = autograd.elementwise_grad(function)

            for i in range(model.number_of_parameters):
                model = model_class()
                default_parameters = lock_unwanted_params(model, model.default_parameters, i)
                upper_bounds = lock_unwanted_params(model, model.upper_bounds, i)
                lower_bounds = lock_unwanted_params(model, model.lower_bounds, i)
                parameter_inds = lock_unwanted_params(model, list(range(model.number_of_parameters)), i)

                for parameter, parameter_name in zip([default_parameters, upper_bounds, lower_bounds], ["default_parameters", "upper_bounds", "lower_bounds"]):
                    if type(parameter[0]) is not float: # only the floats can be differentiable
                        continue
                    try:
                        function(parameter)
                        gradient(parameter)
                    except (Exception, RuntimeWarning) as error:
                        print(f"differentiability errors on model {model_class}, parameter number {i}, parameter set of {parameter_name}\n\t{error}\n")
                        parameter_list = [param._value if hasattr(param, '_value')  else param for param in model.parameters]
                        print(f'model.parameters: {parameter_list}')
                        print(f'parameter: {parameter}, index: {i}')
                        print('\n\n')


def test_differentiability_graphical():
    propagator = Propagator(window_t=1e-9, n_samples=2 ** 14, central_wl=1.55e-6)
    for _, node_sub_classes in config.NODE_TYPES_ALL.items():
        for _, model_class in node_sub_classes.items():
            model = model_class()
            for i in range(0, model.number_of_parameters):
                model = model_class()
                if model.noise_model is None:
                    continue
                if type(model.parameters[i]) == float and model.lower_bounds[i] is not None and model.upper_bounds[i] is not None:
                    _get_gradient_vectors(model, i, model.lower_bounds[i], model.upper_bounds[i], model.default_parameters, propagator, graphical_test='always', noise=True)
        

def _get_gradient_vectors(model, param_index, lower_limit, upper_limit, default_vals, propagator, noise=True, steps=50, rtol=5e-2, atol=1e-9, graphical_test='if failing'):
    """
    Plot gradient vectors. One is computed with autograd, the other with np.diff (finite difference). Also plot the function

    If graphical_test == 'always', displays all plots
    If graphical_test == 'if failing', displays all plots where the gradient and finite difference had a significant difference
    If graphical_test == 'none', does not display plots
    """
    print(f'model: {model.node_acronym}, param: {param_index}')
    AdditiveNoise.simulate_with_noise = noise
    if (lower_limit == upper_limit):
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
                       np.sin(4 * np.pi / propagator.window_t * propagator.t) + np.cos(32 * np.pi / propagator.window_t * propagator.t)) * 0.00001 
        output = model.propagate([input_state], propagator)
        if (model.noise_model is not None):
            output = [model.noise_model.add_noise_to_propagation(output[0], propagator)]

        return np.std(np.abs(output[0])) + np.mean(np.abs(output[0])) + np.std(np.angle(output[0])) * 10
    
    gradient_func = autograd.grad(test_function)

    lock_unwanted_params(model, param_index)
    model.parameters = default_vals

    param_vals, delta = np.linspace(lower_limit, upper_limit, num=steps, retstep=True)
    
    function_output = np.array([test_function(param_val) for param_val in param_vals])
    finite_diff_gradient_output = np.diff(function_output) / delta

    if hasattr(finite_diff_gradient_output, '_value'): # if it's an arraybox object, fix it
        finite_diff_gradient_output = finite_diff_gradient_output._value
    
    gradient_output = np.array([gradient_func(param_val) for param_val in param_vals])
    
    # the gradient is averaged, because the finite differences are closest to the derivative halfway between the parameter values
    # the first two points are ignored, because there's often slightly strange behaviour at 0
    gradients_correct = np.allclose((gradient_output[1:-1] + gradient_output[2:]) / 2, finite_diff_gradient_output[1:], rtol=rtol, atol=atol)
    
    if (not gradients_correct):
        print(f'Calculation mismatch, please inspect plot.')
        # print(f'gradients - finite diff gradients:\n {(gradient_output[1:-1] + gradient_output[2:]) / 2 - finite_diff_gradient_output[1:]}')

    if graphical_test == 'always' or (graphical_test == 'if failing' and not gradients_correct):
        _, ax = plt.subplots(2, 1)
        ax[1].plot(param_vals, gradient_output, label='autograd derivative')
        ax[1].plot(param_vals[0:-1] + delta / 2, finite_diff_gradient_output, label='finite difference derivative', ls='--')
        ax[0].plot(param_vals, function_output, label='function', color='black')
        ax[0].legend()
        ax[0].set_ylabel('f(x)')
        ax[1].set_ylabel('df/dx')
        ax[1].legend()
        plt.title(f'Autograd and finite difference derivatives. Param {param_index}, model {model.node_acronym}, range {lower_limit}-{upper_limit}')
        plt.show()




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

if __name__ == "__main__":
    # unittest.main()
    # test_differentiability()
    # test_differentiability_all_params()
    test_differentiability_graphical()
    # test_distributed_computing()