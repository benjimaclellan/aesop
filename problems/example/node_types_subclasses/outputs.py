"""

"""

import matplotlib.pyplot as plt
import autograd.numpy as np
from scipy.constants import Boltzmann, elementary_charge
from pint import UnitRegistry
unit = UnitRegistry()

from ..node_types import Output

from ..assets.decorators import register_node_types_all
from ..assets.functions import fft_, ifft_, power_
from problems.example.assets.filter import Filter
from problems.example.assets.additive_noise import AdditiveNoise


@register_node_types_all
class MeasurementDevice(Output):
    def __init__(self, **kwargs):
        self.node_lock = True

        self.number_of_parameters = 0
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_sizes = []
        self.parameter_imprecisions = []
        self.parameter_units = []
        self.parameter_locks = []
        self.parameter_names = []

        self.default_parameters = []

        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):
        return states


@register_node_types_all
class Photodiode(Output):
    """
    Assumptions:
        1. Responsivity is (roughly) constant in the relevant frequency range
        2. Also assume all input wavelengths are in an acceptable band
            TODO: reassess this
        3. Ignore max input: simply cap max output (assuming saturation). Print warning at invalid input, however
    TODO:
        1. Cap max power
        2. Can I say: P_in_max * responsivity = I_max = V_max / R_load (get R), then get V_out = I_out * R?
            Basically, is the effective resistance independent of I_max?

    Sources:
    https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=285 (for the bandwidth getting methods)

    Noise sources:
    https://depts.washington.edu/mictech/optics/me557/detector.pdf
    """
    def __init__(self, **kwargs):
        """
        Uses butterworth filter of provided degree
        """
        self.node_lock = True

        self.number_of_parameters = 8
        self.lower_bounds = [0, 1e9, 1, 0, 0, 0, 0, 0]
        self.upper_bounds = [1, 200e9, 20, 0.1, 0.1, 1e3, 320, 1e-6]
        
        self.data_types = ['float'] * self.number_of_parameters
        
        self.step_sizes = [None] * self.number_of_parameters
        self.step_sizes[2] = 1 # order of filter must be integer
        self.parameter_imprecisions = [None] * self.number_of_parameters
        self.parameter_units = [unit.A / unit.W, unit.Hz, None, unit.A, unit.W, unit.ohm, unit.kelvin, unit.A]
        self.parameter_locks = [True] * self.number_of_parameters
        self.parameter_names = ['responsivity', 'bandwidth', 'filter_order', 'max_photocurrent', 'max_power_in', 'load_resistance', 'temp_K', 'dark_current']
        self.default_parameters = [0.5, 100e9, 2, 3e-3, 40e-3, 50, 293, 5e-9] # all default parameters from: https://www.finisar.com/sites/default/files/downloads/xpdv412xr_ultrafast_100_ghz_photodetector_rev_a1_product_specification.pdf

        super().__init__(**kwargs)
        self.filter = Filter(shape='butterworth lowpass', transition_f=self.all_params['bandwidth'], dc_gain=1, order=self.all_params['filter_order'])

        # noise applies onto voltage (could also add it to current, but chose not to)
        # Johnson noise = sqrt(4kTBR), from: https://www.thorlabs.com/images/TabImages/Photodetector_Lab.pdf
        self.noise_model = AdditiveNoise(noise_type='rms constant',
                                         noise_param=np.sqrt(4 * Boltzmann * self.all_params['temp_K'] * \
                                                             self.all_params['bandwidth'] * self.all_params['load_resistance'])) # Johnson noise, assumed to be white and Gaussian
        self.noise_model.add_noise_source(noise_type='shot', # slope of shot noise is 2qBR^2
                                          noise_param=(2 * elementary_charge * self.all_params['bandwidth'] * self.all_params['load_resistance']**2, self.all_params['dark_current']))

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):
        """
        Assumes that if you get multiple input states, they all linearly superpose
        """
        state = states[0]
        for i in range(1, len(states)):
            state = state + states[i]
        
        power_in = power_(state) # ok so this is the power input 
        voltage = self.all_params['responsivity'] * power_in * self.all_params['load_resistance']
        voltage = self.filter.get_filtered_time(voltage, propagator)

        # TODO: implement save_transforms

        # TODO: add noise

        # TODO: clip voltage to max possible voltage (or max photocurrent) (gotta do this in a differentiable way though)

        return [voltage]

    @staticmethod
    def get_bandwidth_from_riseTimeResponse(t):
        return 0.35 / t

    @staticmethod
    def get_bandwidth_from_loadResistance_junctionCapacitance(R, C):
        return 1/ (2 * np.pi * R * C)
