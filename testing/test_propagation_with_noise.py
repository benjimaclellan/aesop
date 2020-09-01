import autograd.numpy as np
import pytest
import random
import matplotlib.pyplot as plt
import seaborn as sns

from autograd import grad

from problems.example.graph import Graph
from problems.example.assets.additive_noise import AdditiveNoise
from problems.example.assets.propagator import Propagator

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice
from problems.example.node_types_subclasses.single_path import CorningFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from lib.analysis.hessian import function_wrapper, get_hessian
from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration


"""
TODO: does it make sense that the osnr parameter doesn't affect the gradient?? Like is there some cancellation
"""


SKIP_GRAPHICAL_TEST = True


@pytest.fixture(scope='function')
def propagator():
    return Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)


@pytest.fixture(scope='function')
def laser_graph():
   return get_laser_graph_osnr(55)


@pytest.fixture(scope='function')
def evaluator(propagator):
    return RadioFrequencyWaveformGeneration(propagator)


@pytest.fixture(scope='function')
def default_graph():
    """
    Returns the default graph for testing, with fixed topology at this time
    """
    return get_default_graph_osnr(55)


def get_laser_graph_osnr(osnr):
    nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':1, 'central_wl':1.55e-6, 'osnr_dB':osnr}),
             1: MeasurementDevice()
            }
    edges = [(0, 1)]
    graph = Graph(nodes, edges, propagate_on_edges=False)
    graph.assert_number_of_edges()
    return graph


def get_default_graph_osnr(osnr):
    """
    Returns the default graph for testing, with fixed topology at this time
    """
    nodes = {0: ContinuousWaveLaser(parameters_from_name={'peak_power': 1, 'central_wl': 1.55e-6, 'osnr_dB':osnr}),
             1: PhaseModulator(parameters_from_name={'depth': 9.87654321, 'frequency': 12e9}),
             2: WaveShaper(),
             3: MeasurementDevice()
            }

    edges = [(0, 1),
             (1, 2),
             (2, 3)]

    graph = Graph(nodes, edges, propagate_on_edges=True)
    graph.assert_number_of_edges()
    return graph


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_laser_graph(laser_graph, propagator):
    laser_graph.propagate(propagator)
    laser_graph.inspect_state(propagator, freq_log_scale=True)

@pytest.mark.xfail
def test_laser_osnr(propagator):
    """
    Tested osnr only laser, but now that linewidth is added that has changed
    """
    for i in range(1, 10):
        graph = get_laser_graph_osnr(i)
        signal = graph.get_output_signal_pure(propagator)
        noise = graph.get_output_noise(propagator)
        print(i)
        print(AdditiveNoise.get_OSNR(signal, noise))
        assert np.isclose(AdditiveNoise.get_OSNR(signal, noise), i)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_default_graph(default_graph, propagator):
    AdditiveNoise.simulate_with_noise = True
    default_graph.propagate(propagator, save_transforms=True)
    default_graph.inspect_state(propagator, freq_log_scale=True)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_default_graph_isolate_noise(default_graph, propagator):
    default_graph.display_noise_contributions(propagator, title='Propagation with laser white noise only')


def test_graph_resampling(laser_graph, propagator):
    # TODO: check resampling works on edges too
    laser_graph.propagate(propagator)
    output0 = np.copy(laser_graph.measure_propagator(laser_graph.get_output_node()))

    laser_graph.resample_all_noise()

    laser_graph.propagate(propagator)
    output1 = laser_graph.measure_propagator(laser_graph.get_output_node())

    assert not np.allclose(output0, output1)


@pytest.mark.skipif(SKIP_GRAPHICAL_TEST, reason='skipping non-automated checks')
def test_propagate_with_without_noise(default_graph, propagator):
    AdditiveNoise.simulate_with_noise = True
    default_graph.propagate(propagator)
    default_graph.inspect_state(propagator)

    AdditiveNoise.simulate_with_noise = False
    default_graph.propagate(propagator)
    default_graph.inspect_state(propagator)


# No tunable parameters in the laser-measurement only graph, so no point in testing grad
@pytest.mark.skip
@pytest.mark.parametrize("graph", [get_default_graph_osnr(55), get_default_graph_osnr(4)])
def test_propagate_autograd_grad(graph, propagator, evaluator):
    # simulate with noise
    AdditiveNoise.simulate_with_noise = True

    NUM_SAMPLES = 100

    graph.propagate(propagator)
    propagate_wrapped = function_wrapper(graph, propagator, evaluator)
    propagate_grad = grad(propagate_wrapped)

    # np.random.seed(0)
    np.random.seed(3)
    params = graph.sample_parameters_to_list() # pick random place from which to sample from

    # show propagation
    if (not SKIP_GRAPHICAL_TEST):
        _, node_edge_index, parameter_index, _, _ = graph.extract_parameters_to_list()
        graph.distribute_parameters_from_list(params, node_edge_index, parameter_index)
        graph.propagate(propagator)
        graph.inspect_state(propagator, freq_log_scale=True)

    grad_sum = np.zeros(len(params))
    for i in range(NUM_SAMPLES):
        graph.resample_all_noise(seed=i)
        grad_eval = propagate_grad(params)
        grad_sum += grad_eval

    average_grad = grad_sum / NUM_SAMPLES

    print(f'\nNoisy grad average over {NUM_SAMPLES} samples:')
    print(average_grad)

    # simulate without noise
    AdditiveNoise.simulate_with_noise = False
    noiseless_grad = propagate_grad(params)
    print('\nNoiseless grad:')
    print(noiseless_grad)

    assert np.allclose(noiseless_grad, average_grad, atol=1e-1)


@pytest.mark.skip
def test_propagate_autograd_hess(default_graph, propagator, evaluator):
    # simulate with noise
    AdditiveNoise.simulate_with_noise = True

    NUM_SAMPLES = 10
    default_graph.propagate(propagator)

    propagate_hess = get_hessian(default_graph, propagator, evaluator)

    np.random.seed(3030)
    params = np.array(default_graph.sample_parameters_to_list())

    hess_sum = np.zeros((len(params), len(params)))
    for i in range(NUM_SAMPLES):
        default_graph.resample_all_noise(seed=i)
        hess_eval = propagate_hess(params)
        hess_sum += hess_eval
    
    average_hess = hess_sum / NUM_SAMPLES

    print(f'\nNoisy hessian average over {NUM_SAMPLES} samples:')
    print(average_hess)

    # simulate without noise
    AdditiveNoise.simulate_with_noise = False
    noiseless_hess = propagate_hess(params)
    print('\nNoiseless hess:')
    print(noiseless_hess)

    if (not SKIP_GRAPHICAL_TEST):
        info = default_graph.extract_attributes_to_list_experimental(['parameters', 'parameter_names'], get_location_indices=True, exclude_locked=True)

        _, ax = plt.subplots(2, 1)
        sns.heatmap(average_hess, ax=ax[0])
        sns.heatmap(noiseless_hess, ax=ax[1])
        for i in range(2):
            ax[i].set(xticks = list(range(len(info['parameters']))), yticks = list(range(len(info['parameters']))))
            ax[i].set_xticklabels(info['parameter_names'], rotation=45, ha='center')
            ax[i].set_yticklabels(info['parameter_names'], rotation=45, ha='right')
        
        plt.show()

    assert np.allclose(noiseless_hess, average_hess, atol=1e-1)
