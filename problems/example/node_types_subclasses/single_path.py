"""

"""
import matplotlib.pyplot as plt
from scipy.constants import Planck, speed_of_light
from math import factorial

from pint import UnitRegistry
unit = UnitRegistry()
import autograd.numpy as np

from ..node_types import SinglePath
from lib.base_classes import NodeType


# from ..assets.decorators import register_node_types_all
from ..assets.decorators import register_node_types_all
from ..assets.functions import fft_, ifft_, psd_, power_, ifft_shift_, dB_to_amplitude_ratio
from ..assets.additive_noise import AdditiveNoise


@register_node_types_all
class DispersiveFiber(SinglePath):
    """

    """
    node_acronym = 'DF'
    def __init__(self, **kwargs):
        self.node_lock = False

        self.number_of_parameters = 1
        self.default_parameters = [1.0]

        self.upper_bounds = [100e3]
        self.lower_bounds = [0.0]
        self.data_types = ['float']
        self.step_sizes = [None]
        self.parameter_imprecisions = [1]
        self.parameter_units = [unit.m]
        self.parameter_locks = [False]
        self.parameter_names = ['length']
        self.parameter_symbols = [r"$x_z$"]
        # self.beta = -26.3e3 * 1e-12 * 1e-12 #fs2/m * s/fs * s/fs
        self._n = 1.44
        self._alpha = 0 # dB/km, from Corning SMF28 datasheet
        # self._alpha = -0.015 # dB/km, from Corning SMF28 datasheet

        self._zdw0 = 1310.0 * 1e-9 #nm -> m (zero-dispersion wavelength)
        # self._S0 = 0.092 * 1e-12 / (1e-9 * 1e-9 * 1e3)# zero-dispersion slope, ps/(nm2 * km) -> s/m^3
        self._S0 = -0.155 * 1e-12 / (1e-9 * 1e-9 * 1e3)# zero-dispersion slope, ps/(nm2 * km) -> s/m^3
        # self._S0 = -0.8 * 1e-12 / (1e-9 * 1e-9 * 1e3) # zero-dispersion slope, ps/(nm2 * km) -> s/m^3


        self._beta2_experimental = -22 * 1e-12 * 1e-12 / (1e3)  # ps^2/(km)  # standard SMF chromatic dispersion
        super().__init__(**kwargs)
        return

    # TODO: initialize any large-ish variables/arrays that don't change for each component model (i.e. frequency arrays)
    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators
        length = self.parameters[0]

        # _lambda0 = speed_of_light/((propagator.central_frequency))
        # _lambda = speed_of_light/((propagator.f + propagator.central_frequency))
        #
        # D_lambda = self._S0 / 4.0 * (_lambda - self._zdw0 ** 4.0 / _lambda ** 3.0)
        # beta_2 = _lambda**2.0 * D_lambda / (-2.0 * np.pi * speed_of_light)

        # beta_2 = self._beta2_experimental # simple description
        beta_2 = -22 * 1e-12 * 1e-12 / (1e3) # simple description
        propagation_constant = length * beta_2 * np.power(2 * np.pi * (propagator.f), 2)
        if save_transforms:
            self.transform = (('f', D_lambda, r'D(lambda)'),)
        else:
            self.transform = None

        state = ifft_( ifft_shift_(np.exp(1j * (propagation_constant)), ax=0) * fft_(state, propagator.dt), propagator.dt)
        state = state * dB_to_amplitude_ratio(self._alpha * length/1000.0)
        return state

@register_node_types_all
class VariableOpticalAttenuator(SinglePath):
    """

    """
    node_acronym = 'VOA'

    def __init__(self, **kwargs):
        self.node_lock = False

        self.number_of_parameters = 1
        self.default_parameters = [0.0]

        self.upper_bounds = [0.0]
        self.lower_bounds = [-60.0]
        self.data_types = ['float']
        self.step_sizes = [None]
        self.parameter_imprecisions = [1]
        self.parameter_units = [unit.dB]
        self.parameter_locks = [False]
        self.parameter_names = ['attenuation']
        self.parameter_symbols = [r"$x_\alpha$"]

        super().__init__(**kwargs)
        return

    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators
        attenuation = self.parameters[0]

        if save_transforms:
            self.transform = None
        else:
            self.transform = None

        state = state * dB_to_amplitude_ratio(attenuation)
        return state

@register_node_types_all
class PhaseModulator(SinglePath):
    """
    """
    node_acronym = 'PM'

    max_frequency = 50.0e9
    min_frequency = 1.0e9
    step_frequency = 1.0e9

    def __init__(self, phase_noise_points=None, **kwargs):
        # fixing this up, because phase noise points isn't a proper parameter, just a way to GET the proper param
        if phase_noise_points is not None:
            FWHM = AdditiveNoise.get_estimated_FWHM_linewidth_from_points(phase_noise_points)
            if 'parameters' in kwargs:
                kwargs['parameters'][3] = FWHM
            elif 'parameters_from_name' in kwargs:
                kwargs['parameters_from_name']['FWHM_linewidth'] = FWHM

        self.node_lock = False

        self.number_of_parameters = 4
        self.default_parameters = [1.0, 12.0e9, 0.01, 0.0]

        self.upper_bounds = [4*np.pi, self.max_frequency, 2*np.pi, 1e9]
        self.lower_bounds = [0.001, self.min_frequency, 0.01, 0.0]
        self.data_types = ['float', 'float', 'float', 'float']
        self.step_sizes = [None, self.step_frequency, None, None]
        self.parameter_imprecisions = [1.0, 1.0, 0.1, 1.0]
        self.parameter_units = [unit.rad, unit.Hz, unit.rad, unit.Hz]
        self.parameter_locks = [False, False, False, True]
        self.parameter_names = ['depth', 'frequency', 'shift', 'FWHM_linewidth']
        self.parameter_symbols = [r"$x_m$", r"$x_f$", r"$x_s$", r"$x_{FWHM}$"]

        self._loss_dB = -4.0 # dB

        super().__init__(**kwargs)

        """
        General noise guidelines:
        If RF signal generator phase noise characteristics are given as % error of frequency or ppm, use: "frequency ppm" noise type
        (calculates value of % error, uses it as linewidth)
        See: https://www.jitterlabs.com/support/calculators/ppm
        
        If RF signal generator phase noise characteristics are given as phase noise at offset from carrier, use 
        (list of points (offset, dBc/Hz), which are fitted to the curve C/f^2, where C = h0/2 and FWHM = pi * h0)

        Else, specify as actual linewidth "FWHM linewidth". If you want to unlock a parameter and optimize it, you SHOULD use linewidth. 
        
        Note: all noise metrics representation whould be FWHM linewidth, under the assumption that linewidth is Lorentzian/
        """
        self.update_noise_model()

    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators
        depth = self.parameters[0]
        frequency = self.parameters[1]

        transform = depth * (np.cos(2 * np.pi * frequency * propagator.t + self.noise_model.get_phase_noise(propagator)))
        if save_transforms:
            self.transform = (('t', transform, 'modulation'),)
        else:
            self.transform = None

        state = state * np.exp(1j * transform)
        state = state * dB_to_amplitude_ratio(self._loss_dB)
        return state
    
    def update_noise_model(self):
        self.noise_model = AdditiveNoise(noise_type='phase noise from linewidth', noise_param=self.parameters[3], noise_on=False)

@register_node_types_all
class IntensityModulator(SinglePath):
    """
https://www.lasercomponents.com/fileadmin/user_upload/home/Datasheets/lc/application-reports/ixblue/introduction-to-modulator-bias-controllers.pdf
    """
    node_acronym = 'IM'
    max_frequency = 50.0e9
    min_frequency = 1.0e9
    step_frequency = 1.0e9

    def __init__(self, **kwargs):
        self.node_lock = False

        self.number_of_parameters = 4
        self.default_parameters = [1.0, 1.0e9, 0.0, 0.0]

        self.upper_bounds = [2*np.pi, 50.0e9, 2 * np.pi, 2*np.pi]
        self.lower_bounds = [0.0, 1.0e9, 0.0, 0.0]
        self.data_types = ['float', 'float', 'float', 'float']
        self.step_sizes = [None, 1e9, None, None]
        self.parameter_imprecisions = [1.0, 10e6, 0.1, 0.1]
        self.parameter_units = [unit.rad, unit.Hz, unit.rad, unit.rad]
        self.parameter_locks = [False, False, False, False]
        self.parameter_names = ['depth', 'frequency', 'bias', 'shift']
        self.parameter_symbols = [r"$x_m$", r"$x_f$", r"$x_{DC}$", "r$x_{\phi}$"]

        self._loss_dB = -4.0 # dB

        super().__init__(**kwargs)

        return

    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators

        depth = self.parameters[0]
        frequency = self.parameters[1]
        bias = self.parameters[2]
        # shift = self.parameters[3]

        transform = depth * (np.cos(2 * np.pi * frequency * propagator.t)) + bias

        if save_transforms:
            self.transform = (('t', power_( np.ones_like(state)/2.0 + np.ones_like(state) / 2.0 * np.exp(1j * transform)), 'modulation'),)
        else:
            self.transform = None
        state = state/2.0 + state / 2.0 * np.exp(1j * transform)
        state = state * dB_to_amplitude_ratio(self._loss_dB)

        return state


@register_node_types_all
class WaveShaper(SinglePath):
    """

    """

    number_of_bins = 15
    frequency_bin_width = 12e9
    extinction_ratio = 10 ** (-35 / 10)
    node_acronym = 'WS'

    def __init__(self, **kwargs):

        number_of_bins = self.number_of_bins

        self.node_lock = False
        self.number_of_parameters = 2 * number_of_bins

        self.default_parameters = [1.0] * number_of_bins + [0.1] * number_of_bins

        self.upper_bounds = [1.0] * number_of_bins + [2*np.pi] * number_of_bins
        self.lower_bounds = [self.extinction_ratio] * number_of_bins + [0.0] * number_of_bins
        self.data_types = 2 * number_of_bins * ['float']
        self.step_sizes = [None] * number_of_bins + [None] * number_of_bins
        self.parameter_imprecisions = [0.1] * number_of_bins + [0.1 * 2 * np.pi] * number_of_bins
        self.parameter_units = [None] * number_of_bins + [unit.rad] * number_of_bins
        self.parameter_locks = 2 * number_of_bins * [False]
        self.parameter_names = ['amplitude{}'.format(ind) for ind in range(number_of_bins)] + \
                               ['phase{}'.format(ind) for ind in range(number_of_bins)]
        self.parameter_symbols = [r"$x_{a_{"+"{:+d}".format(ind)+r"}}$" for ind in range(-(number_of_bins-1)//2, (number_of_bins-1)//2+1)] + \
                                 [r"$x_{\phi_{"+"{:+d}".format(ind)+r"}}$" for ind in range(-(number_of_bins-1)//2, (number_of_bins-1)//2+1)]

        self._loss_dB = -4.5 # dB

        super().__init__(**kwargs)
        return

    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators

        # Slice at into the first half (amp) and last half (phase)
        amplitudes = self.parameters[:self.number_of_bins]
        phases = self.parameters[self.number_of_bins:]

        N = np.shape(propagator.f)[0]
        n = np.floor(propagator.n_samples / ((1 / propagator.dt) / self.frequency_bin_width)).astype('int')
        tmp = np.ones((n, 1))

        a = np.array([i * tmp for i in amplitudes])
        p = np.array([i * tmp for i in phases])

        amp1 = np.concatenate(a)
        phase1 = np.concatenate(p)


        left = np.floor((propagator.n_samples - amp1.shape[0]) / 2).astype('int')
        right = propagator.n_samples - np.ceil((propagator.n_samples - amp1.shape[0]) / 2).astype('int')

        if left < 0:
            amplitude_mask = amp1[-left:right, :]
            phase_mask = phase1[-left:right, :]
        else:
            # we will pad amp1 and phase1 with zeros so they are the correct size
            pad_left = np.ones((left, 1)) * self.extinction_ratio
            pad_right = np.ones((N - right, 1)) * self.extinction_ratio

            # Concatenate the arrays together
            # We cannot use array assignment as it is not supported by autograd
            amplitude_mask = np.concatenate((pad_left, amp1, pad_right), axis=0)
            phase_mask = np.concatenate((pad_left, phase1, pad_right), axis=0)

        state = ifft_(ifft_shift_(amplitude_mask * np.exp(1j * phase_mask), ax=0) * fft_(state, propagator.dt), propagator.dt)
        state = state * dB_to_amplitude_ratio(self._loss_dB)

        if save_transforms:
            self.transform = (('f', amplitude_mask, 'amplitude'), ('f', phase_mask, 'phase'),)
        else:
            self.transform = None

        return state


# @register_node_types_all
class IntegratedSplitAndDelayLine(SinglePath):
    """
    """
    node_acronym = 'SDL'
    number_of_parameters = 8
    def __init__(self, **kwargs):
        self.node_lock = False


        self.default_parameters = [0.1] * self.number_of_parameters

        self.upper_bounds = [1.0] * self.number_of_parameters
        self.lower_bounds = [0.0] * self.number_of_parameters
        self.data_types = ['float'] * self.number_of_parameters
        self.step_sizes = [None] * self.number_of_parameters
        self.parameter_imprecisions = [0.01] * self.number_of_parameters
        self.parameter_units = [None] * self.number_of_parameters
        self.parameter_locks = [False] * self.number_of_parameters
        self.parameter_names = ['coupling_ratio{}'.format(ind) for ind in range(self.number_of_parameters)]
        self.parameter_symbols = [r"$x_{{"+f"{ind}"+r"}}$" for ind in range(self.number_of_parameters)]

        self._n = 1.444
        self._delays = [2**k * 1e-12 for k in range(0, self.number_of_parameters)]

        self._loss_dB = 1.0 # dB

        # [1e-12, 2e-12, 4e-12, 8e-12, 16e-12, 32e-12, 128e-12, 64e-12

        super().__init__(**kwargs)
        return

    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators
        coupling_ratios = self.parameters

        # field is in the spectral domain here now
        field_short = fft_(state, propagator.dt)
        field_long = np.zeros_like(field_short)

        field_short_tmp = np.zeros_like(field_short)

        for (coupling_ratio, delay) in zip(coupling_ratios, self._delays):
            length = (propagator.speed_of_light / self._n) * delay
            beta = self._n * (2 * np.pi * (propagator.f + propagator.central_frequency)) / propagator.speed_of_light

            field_short_tmp = field_short

            try:
                short_coupling, long_coupling = np.cos(np.pi/2 * coupling_ratio), np.sin(np.pi/2 * coupling_ratio)

                field_short = (short_coupling * field_short + 1j * long_coupling * field_long)
                field_long = np.exp(1j * ifft_shift_(beta * length, ax=0)) * (
                        1j * long_coupling * field_short_tmp + short_coupling * field_long)
            except RuntimeWarning as w:
                print(f'RuntimeWarning: {w}')
                print(f'coupling ratios: {coupling_ratios}')
                print(f'coupling ratio: {coupling_ratio}')
                raise w

        if save_transforms:
            transform = np.zeros_like(propagator.t).astype('float')
            for (coupling_ratio, delay) in zip(coupling_ratios, self._delays):
                transform[(np.abs(propagator.t - (-delay))).argmin()] = coupling_ratio
            self.transform = (('t', transform, 'delay coupling ratios'),)
        else:
            self.transform = None

        return ifft_(field_short, propagator.dt) * dB_to_amplitude_ratio(self._loss_dB)


@register_node_types_all
class OpticalAmplifier(SinglePath):
    """
    EDFA modelled as follows:

    ---------------g(f), the small-signal gain---------------
    
    In the following equations, f = peak frequency - actual frequency
    g(f) = g_max * exp(-f^2/Beta)
    With Beta = d^2/(ln(10^(G_f / 10))),
    d = max distance between peak wavelength and the edge of the band,
    Gf = gain flatness in dB,

    Such that, as expected, g(d) = g_min

    --------------- G(f), the true gain---------------
    Defined according to this model:
    https://www.researchgate.net/publication/3289918_Simple_black_box_model_for_erbium-doped_fiber_amplifiers [1]

    TODO: figure out how to handle too powerful input signals (which are 'invalid inputs')
    """
    node_acronym = 'EDFA'
    number_of_parameters = 9

    def __init__(self, **kwargs):
        """
        Possible values to include in kwargs

        :param max_small_signal: maximum small signal gain, in dB (assumed to be at band center)
        :param peak_wl: peak wavelength of the EDFA where max amplification occurs, in m (max amplification there)
        :param band_lower: min wavelength, in m
        :param band_upper: max wavelength, in m
        :param P_out_max: maximum output power, in dBm
        :param gain_flatness : gain flatness, in dB (Gmax(dB) - Gmin(dB))
        :param alpha : parameter alpha, as defined in [1] (default value of 1 seems reasonable)
        """
        self.node_lock = False

        self.upper_bounds = [50.0, 1612e-9, 1600e-9, 1625e-9, 10.0, 10.0, 15.0, 1.5, 10.0]
        self.lower_bounds = [1.0, 1535e-9, 1530e-9, 1540e-9, 1e-4, 1e-7, 0.0, 0.0, 3.0]
        self.data_types = ['float'] * self.number_of_parameters
        self.step_sizes = [None] * self.number_of_parameters
        self.parameter_imprecisions = [1, 1e-9, 1e-9, 1e-9, 1e-3, 1e-3, 1, 0.1, 1] # placeholders, I don't really know
        self.parameter_units = [None, unit.m, unit.m, unit.m, unit.W, unit.W, None, None, None] # no dB unit exists in registry
        self.parameter_locks = [False] + [True] * (self.number_of_parameters - 1)
        self.parameter_names = ['max_small_signal_gain_dB', 'peak_wl', 'band_lower', 'band_upper', 'P_in_max', 'P_out_max', 'gain_flatness_dB', 'alpha', 'max_noise_fig_dB']

        self.default_parameters = [30.0, 1550e-9, 1520e-9, 1565e-9, 0.01, 0.1, 1.5, 1.0, 5.0]
        self.parameter_symbols = [r"$x_{{"+f"{ind}"+r"}}$" for ind in range(self.number_of_parameters)]

        super().__init__(**kwargs)
        self.noise_model = AdditiveNoise(noise_type='edfa ASE', noise_param=self)

        # save for noise propagation
        self._last_gain = None
        self._last_noise_factor = None

        #
        self.set_parameters_as_attr()

    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators
        """
        """
        self.set_parameters_as_attr()

        state_f = fft_(state, propagator.dt)

        gain = self._gain(state, propagator)

        self._noise_factor(state, propagator) # saves it for the noise call

        if save_transforms:
            self.transform = (('f', 10 * np.log10(ifft_shift_(np.sqrt(gain))), 'gain (dB)'),)

        return ifft_(np.sqrt(gain) * state_f, propagator.dt) # sqrt of gain, because gain is a ratio of power, not amplitude
    
    def get_ASE_filter(self, propagator):
        """
        NOTE: this method must be called after the corresponding propagate call. 
        Returns spectral density of ASE as P_ase = (G * F - 1) * h * v

        (1) https://perso.telecom-paristech.fr/gallion/documents/free_downloads_pdf/PG_revues/PG_R66.pdf
        (2) http://notes-application.abcelectronique.com/018/18-27242.pdf
        (3) https://www.hft.tu-berlin.de/fileadmin/fg154/ONT/Skript/ENG-Ver/EDFA.pdf

        Note that we want to take into account the bandwidth only of the detector used to characterise

        Removed factor of 2 from (1) because our signal does not care about polarization

        Return ASE shape, ASE total power (this is so that the noise is still randomized rather than deterministic)
        Assume the bandwidth of the detector used to find the noise figure is
        """
        FG = self._last_noise_factor * self._last_gain # properly should be FG - 1, if we account for signal shot, but this is very close except at low G
        expected_power_dist = Planck * (ifft_shift_(propagator.f) + propagator.central_frequency) / 2 * \
            FG * propagator.df

        return np.sqrt(expected_power_dist)

    def _gain(self, state, propagator):
        """
        Gain is defined as in [1], G = g / (1 + (g * P_in / P_max)^alpha), with g = small signal gain, G is true gain
        """
        small_signal_gain = self._get_small_signal_gain(propagator)

        P_in = np.mean(power_(state)) # EDFAs saturation is affected by average power according to

        # if P_in > self._P_in_max:
        #     raise ValueError(f'input signal {P_in} is greater than max input signal {self._P_in_max}')

        self._last_gain = small_signal_gain / (1 + (small_signal_gain * P_in / self._P_out_max)**self._alpha) 
        return self._last_gain
    
    def _noise_factor(self, state, propagator):
        """
        TODO: implement fancy noise factor explained here (or decide whether it's worth it: it wouldn't be given by specs)
        F = F_0 + k1 * exp(k2 (G_0(dB) - G(dB)))

        F = noise factor
        F_0 = small signal noise factor
        k1, k2 are exponents. k2 = 0.2, for simplicity, k1 is deduced from the desired 'max' NF (at 3 dB)
        """
        NF_max = self._max_noise_fig_dB
        self._last_noise_factor = 10**(NF_max / 10)
        return self._last_noise_factor

    def _get_small_signal_gain(self, propagator):
        """
        From above...
        ---------------g(f), the small-signal gain---------------
            
            In the following equations, f = peak frequency - actual frequency
            g(f) = g_max * exp(-f^2/Beta)
            With Beta = d^2/(ln(10^(G_f / 10))),
            d = max difference in frequency between the centre of the band and the edge,
            Gf = gain flatness in dB,

            Such that, as expected, g(d) = g_min
        """
        g = 10**(self._max_small_signal_gain_dB / 10)
        
        if np.isclose(self._gain_flatness_dB, 0, atol=1e-12):
            return np.ones_like(propagator.f) * g # if flatness of 0, the gain is constant over all frequencies
    
        central_freq = speed_of_light / self._peak_wl
        lower_freq = speed_of_light / self._band_upper
        upper_freq = speed_of_light / self._band_lower
        d = np.maximum(central_freq - lower_freq, upper_freq - central_freq)

        beta = d**2 / (np.log(10**(self._gain_flatness_dB / 10)))
        f = (ifft_shift_(propagator.f) + propagator.central_frequency) - central_freq

        return g * np.exp(-1 * np.power(f, 2) / beta)


    @staticmethod
    def _calculate_small_signal_gain(peak_wl, signal_central_freq, band_upper, band_lower, gain_flatness, peak_small_signal_gain):
        central_freq = speed_of_light / peak_wl
        lower_freq = speed_of_light / band_upper
        upper_freq = speed_of_light / band_lower
        d = np.maximum(central_freq - lower_freq, upper_freq - central_freq)

        beta = d**2 / (np.log(10**(gain_flatness / 10)))
        f = signal_central_freq - central_freq  # the further the propagator frequency is from the central frequency, the lesser the small signal gain
        g = 10**(peak_small_signal_gain/ 10)

        return g * np.exp(-1 * np.power(f, 2) / beta)


    def display_small_signal_gain(self):
        _, ax = plt.subplots()
        wavelengths = [i*1e-9 for i in range(self._band_lower, self._band_upper)]
        gain = np.array([self._calculate_small_signal_gain(self._peak_wl,
                                                           speed_of_light / w,
                                                           self._band_upper,
                                                           self._band_lower,
                                                           self._gain_flatness_dB,
                                                           self._max_small_signal_gain_dB) for w in wavelengths])
        ax.plot(np.array(wavelengths) * 1e9, gain)
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('gain (ratio)')
        plt.title(f"Small signal gain vs. wavelength \
            \ngain: {self._max_small_signal_gain_dB} dB \
            \npeak: {self._peak_wl * 1e9} nm \
            \nband: {self._band_lower * 1e9}-{self._band_upper * 1e9} nm \
            \ngain flatness: {self._gain_flatness_dB} dB")
        plt.show()
    
    def display_gain(self, states, propagator):
        _, ax = plt.subplots()
        for _, state in enumerate(states):
            gain = self._gain(state, propagator)
            ax.plot(propagator.f, np.fft.fftshift(gain, axes=0), label=f'power: {np.mean(power_(state))}')
        
        ax.legend()
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (ratio)')
        plt.title('Gain, as function of offset frequency and input power')
        
        # plt.title(f"Gain\
        #     \ngain: {self._max_small_signal_gain_dB} dB \
        #     \npeak: {self._peak_wl* 1e9} nm \
        #     \nband: {self._band_lower * 1e9}-{self._band_upper * 1e9} nm \
        #     \ngain flatness: {self._gain_flatness_dB} dB \
        #     \nmax power: {self._P_out_max}")
        plt.show()


@register_node_types_all
class ProgrammableFilter(SinglePath):
    node_acronym = 'PF'

    number_of_bases = 10
    _number_of_bases = number_of_bases
    band_width = 5e12
    extinction_ratio = 10 ** (-35 / 10)
    number_of_parameters = 2 * number_of_bases
    def __init__(self, **kwargs):
        self.node_lock = False

        number_of_bases = self.number_of_bases
        self.default_parameters = [1.0] * number_of_bases + [0.0] * number_of_bases

        self.upper_bounds = [1.0] * number_of_bases + [2 * np.pi] * number_of_bases
        self.lower_bounds = [0.0] * number_of_bases + [0.0] * number_of_bases
        self.data_types = 2 * number_of_bases * ['float']
        self.step_sizes = [None] * number_of_bases + [None] * number_of_bases
        self.parameter_imprecisions = [0.1] * number_of_bases + [0.1 * 2 * np.pi] * number_of_bases
        self.parameter_units = [None] * number_of_bases + [unit.rad] * number_of_bases
        self.parameter_locks = self.number_of_parameters * [False]
        self.parameter_names = ['amplitude{}'.format(ind) for ind in range(number_of_bases)] + \
                               ['phase{}'.format(ind) for ind in range(number_of_bases)]
        self.parameter_symbols = [r"$x_{b_{" + "{:+d}".format(ind) + r"}}$" for ind in
                                  range(-(number_of_bases - 1) // 2, (number_of_bases - 1) // 2 + 1)] + \
                                 [r"$x_{p_{" + "{:+d}".format(ind) + r"}}$" for ind in
                                  range(-(number_of_bases - 1) // 2, (number_of_bases - 1) // 2 + 1)]

        self._loss_dB = -4.5 # dB

        super().__init__(**kwargs)
        return

    @staticmethod
    def bernstein(t, i=0, n=0):
        b = (factorial(n) / (factorial(i) * factorial(n - i))) * np.power(t, i) * np.power(1 - t, n - i)
        return b

    def get_waveform(self, t, coeffs):
        y = np.zeros_like(t)
        for m, coeff in enumerate(coeffs):
            y += coeff * self.bernstein(t, i=m, n=len(coeffs) - 1)
        return y

    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators

        # Slice at into the first half (amp) and last half (phase)
        bernstein_amp_coeffs = self.parameters[:self._number_of_bases]
        bernstein_phase_coeffs = self.parameters[self._number_of_bases:]

        bw_int = int(round(self.band_width/2/propagator.df))

        mask = np.zeros_like(propagator.f)
        mask[np.abs(propagator.f) <= self.band_width / 2] = 1

        f_shifted = np.expand_dims(np.linspace(-propagator.n_samples / 2 - bw_int,
                                 propagator.n_samples / 2 - bw_int,
                                 propagator.n_samples) / (2 * bw_int) + 1, 1)

        amplitude_mask = self.get_waveform(f_shifted, bernstein_amp_coeffs) * mask
        phase_mask = self.get_waveform(f_shifted, bernstein_phase_coeffs) * mask

        state = ifft_(ifft_shift_(amplitude_mask * np.exp(1j * phase_mask), ax=0) * fft_(state, propagator.dt),
                      propagator.dt)
        state = state * dB_to_amplitude_ratio(self._loss_dB)

        if save_transforms:
            self.transform = (('f', amplitude_mask, 'amplitude'), ('f', phase_mask, 'phase'),)
        else:
            self.transform = None
        return state


@register_node_types_all
class DelayLine(SinglePath):
    """
    """
    node_acronym = 'DL'
    number_of_parameters = 1
    def __init__(self, **kwargs):
        self.node_lock = False

        self.default_parameters = [1.0e-9]

        self.upper_bounds = [5.0e-9]
        self.lower_bounds = [0.0]
        self.data_types = ['float']
        self.step_sizes = [1e-9]
        self.parameter_imprecisions = [10.0e-12]
        self.parameter_units = [None]
        self.parameter_locks = [False]
        self.parameter_names = ['delay']
        self.parameter_symbols = [r"$x_{d}$"]

        self._loss_dB = -0.0 # dB

        self._n = 1.444

        super().__init__(**kwargs)
        return

    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators
        delay = self.parameters[0]

        length = (propagator.speed_of_light / self._n) * delay
        beta = self._n * (2 * np.pi * (propagator.f + propagator.central_frequency)) / propagator.speed_of_light

        state = ifft_(np.exp(1j * ifft_shift_(beta * length, ax=0)) * fft_(state, propagator.dt), propagator.dt)
        state = state * dB_to_amplitude_ratio(self._loss_dB)

        if save_transforms:
            transform = np.zeros_like(propagator.t).astype('float')
            transform[(np.abs(propagator.t - (-delay))).argmin()] = 1
            self.transform = (('t', transform, 'delay'),)
        else:
            self.transform = None

        return state




class PhaseShifter(NodeType):
    """
    This will be used in ASOPE to search for 'sensing' setups, i.e. ones for which an objective function has the largest
    sensitivity to a phase shift (i.e. large first-order derivative w.r.t. this phase value)
    """
    node_acronym = 'PS'
    number_of_parameters = 1

    def __init__(self, **kwargs):
        self.node_lock = True
        self._node_type = "signal node"

        self._range_input_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children
        self._range_output_edges = (1, 1)  # minimum, maximum number of input edges, may be changed in children


        self.default_parameters = [0.5*np.pi]

        self.upper_bounds = [+1.0*np.pi]
        self.lower_bounds = [-1.0*np.pi]
        self.data_types = ['float']
        self.step_sizes = [None]
        self.parameter_imprecisions = [0.1*np.pi]
        self.parameter_units = [unit.rad]
        self.parameter_locks = [True]
        self.parameter_names = ['phase']
        self.parameter_symbols = [r"$x_{\phi}$"]

        self._loss_dB = -0.0 # dB

        self._n = 1.444

        super().__init__(**kwargs)
        return

    def propagate(self, state, propagator, save_transforms=False):  # node propagate functions always take a list of propagators

        phase = self.parameters[0]

        delay = propagator.central_wl / (2.0 * np.pi) * phase
        length = (propagator.speed_of_light / self._n) * delay
        beta = self._n * (2 * np.pi * (propagator.f + propagator.central_frequency)) / propagator.speed_of_light
        state = ifft_(np.exp(1j * ifft_shift_(beta * length, ax=0)) * fft_(state, propagator.dt), propagator.dt)
        state = state * dB_to_amplitude_ratio(self._loss_dB)
        if save_transforms:
            transform = np.zeros_like(propagator.t).astype('float')
            transform[(np.abs(propagator.t - (-delay))).argmin()] = 1
            self.transform = (('t', transform, 'delay'),)
        else:
            self.transform = None
        return state