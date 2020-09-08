import autograd.numpy as np

from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, EDFA
from problems.example.assets.additive_noise import AdditiveNoise

np.random.seed(0)

# setup graph
propagator = Propagator(window_t = 1e-9, n_samples = 2**14, central_wl=1.55e-6)
nodes = {0:ContinuousWaveLaser(parameters_from_name={'peak_power':0.01, 'central_wl':1.55e-6, 'FWHM_linewidth':1e8}),
          1:PhaseModulator(parameters_from_name={'depth':9.87654321, 'frequency':12e9}),
          2:WaveShaper(),
          3:Photodiode(parameters_from_name={'filter_order':2})}
        
edges = [(0,1), (1,2), (2,3)]


graph = Graph(nodes, edges, propagate_on_edges=False)
graph.assert_number_of_edges()

# run with noise on
AdditiveNoise.simulate_with_noise = True # this is set by default, but line is included for completeness
graph.propagate(propagator)
graph.inspect_state(propagator, freq_log_scale=True)
graph.display_noise_contributions(propagator) # always shows the noise, regardless of external settings

# run with noise off
AdditiveNoise.simulate_with_noise = False # this is set by default, but line is included for completeness
graph.propagate(propagator)
graph.inspect_state(propagator, freq_log_scale=True)
