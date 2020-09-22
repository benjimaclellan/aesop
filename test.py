
import unittest
import copy
import psutil
import ray
import autograd.numpy as np
import autograd
import warnings

import config.config as config

from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.evolver import Evolver
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter


class Test(unittest.TestCase):
    def test_all_available_nodes(self):
        propagator = Propagator(window_t=100e-12, n_samples=2 ** 14)
        for node_type, nodes in configuration.NODE_TYPES_ALL.items():
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
    test_differentiability()
    # test_distributed_computing()