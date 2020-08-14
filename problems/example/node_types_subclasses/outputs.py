"""

"""

import matplotlib.pyplot as plt
import autograd.numpy as np
from pint import UnitRegistry
unit = UnitRegistry()

from ..node_types import Output

from ..assets.decorators import register_node_types_all
from ..assets.functions import fft_, ifft_, power_
from problems.example.assets.filter import Filter

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
    """
    def __init__(self, **kwargs):
        """
        Uses butterworth filter of provided degree
        """
        self.node_lock = True

        self.number_of_parameters = 6
        self.lower_bounds = [0, 1e9, None, 0, 0, 0]
        self.upper_bounds = [1, 200e9, None, 0.1, 0.1, 1e3]
        
        self.data_types = ['float'] * self.number_of_parameters
        self.data_types[2] = 'string'
        
        self.step_sizes = [None] * self.number_of_parameters
        self.parameter_imprecisions = [None] * self.number_of_parameters
        self.parameter_units = [unit.A / unit.W, unit.Hz, None, unit.A, unit.W, unit.ohm]
        self.parameter_locks = [True] * self.number_of_parameters
        self.parameter_names = ['responsivity', 'bandwidth', 'filter_order', 'max_photocurrent', 'max_power_in', 'load_resistance']
        self.default_parameters = [0.5, 100e9, 2, 3e-3, 40e-3, 50] # all default parameters from: https://www.finisar.com/sites/default/files/downloads/xpdv412xr_ultrafast_100_ghz_photodetector_rev_a1_product_specification.pdf

        super().__init__(**kwargs)
        self.filter = Filter(shape='butterworth lowpass', transition_f=self.all_params['bandwidth'], dc_gain=1, order=self.all_params['filter_order'])

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

        return [voltage]

    @staticmethod
    def get_bandwidth_from_riseTimeResponse(t):
        return 0.35 / t

    @staticmethod
    def get_bandwidth_from_loadResistance_junctionCapacitance(R, C):
        return 1/ (2 * np.pi * R * C)
