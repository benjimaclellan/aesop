"""

"""
import matplotlib.pyplot as plt
from scipy.constants import Planck, speed_of_light
from math import factorial

from pint import UnitRegistry
unit = UnitRegistry()
import autograd.numpy as np

from ..node_types import SinglePath

from ..assets.decorators import register_node_types_all
from ..assets.functions import fft_, ifft_, psd_, power_, ifft_shift_
from ..assets.additive_noise import AdditiveNoise


@register_node_types_all
class CorningFiber(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'DF'

        self.number_of_parameters = 1
        self.default_parameters = [1.0]

        self.upper_bounds = [100e3]
        self.lower_bounds = [0.01]
        self.data_types = ['float']
        self.step_sizes = [None]
        self.parameter_imprecisions = [1]
        self.parameter_units = [unit.m]
        self.parameter_locks = [False]
        self.parameter_names = ['length']
        self.parameter_symbols = [r"$x_\beta$"]
        self.beta = 1e-26
        self._n = 1.44

        super().__init__(**kwargs)
        return

    # TODO: initialize any large-ish variables/arrays that don't change for each component model (i.e. frequency arrays)
    # TODO : check this, and every other model for correctness (so far its been about logic flow)
    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]
        length = self.parameters[0]
        b2 = length * self.beta * np.power(2 * np.pi * propagator.f, 2)

        b1 = length * self._n * (2 * np.pi * (propagator.f + propagator.central_frequency)) / propagator.speed_of_light

        if save_transforms:
            self.transform = (('f', b1, 'b1'), ('f', b2, 'b2'),)
        else:
            self.transform = None

        state = ifft_( ifft_shift_(np.exp(1j * (b1 + b2)), ax=0) * fft_(state, propagator.dt), propagator.dt)
        return [state]

@register_node_types_all
class VariableOpticalAttenuator(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'VOA'

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

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]
        attenuation = self.parameters[0]

        if save_transforms:
            self.transform = None
        else:
            self.transform = None

        state = state * np.exp(0.5 * attenuation)
        return [state]

@register_node_types_all
class PhaseModulator(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'PM'

        self.number_of_parameters = 3
        self.default_parameters = [1.0, 12.0e9, 0.0]

        self.upper_bounds = [np.pi, 48.0e9, 2*np.pi]
        self.lower_bounds = [0.001, 2.0e9, 0.01]
        self.data_types = ['float', 'float', 'float']
        self.step_sizes = [None, 2e9, None]
        self.parameter_imprecisions = [1.0, 1.0, 0.1]
        self.parameter_units = [unit.rad, unit.Hz, unit.rad]
        self.parameter_locks = [False, False, False]
        self.parameter_names = ['depth', 'frequency', 'shift']
        self.parameter_symbols = [r"$x_m$", r"$x_f$", r"$x_s$"]
        super().__init__(**kwargs)

        return

    def propagate(self, states, propagator, num_inputs = 1, num_outputs = 0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]

        depth = self.parameters[0]
        frequency = self.parameters[1]

        transform = depth * (np.cos(2 * np.pi * frequency * propagator.t))
    
        if save_transforms:
            self.transform = (('t', transform, 'modulation'),)
        else:
            self.transform = None

        state1 = state * np.exp(1j * transform)
        return [state1]


@register_node_types_all
class IntensityModulator(SinglePath):
    """
https://www.lasercomponents.com/fileadmin/user_upload/home/Datasheets/lc/application-reports/ixblue/introduction-to-modulator-bias-controllers.pdf
    """

    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'IM'

        self.number_of_parameters = 3
        self.default_parameters = [1.0, 12.0e9, 0.0]

        self.upper_bounds = [np.pi, 48.0e9, 2 * np.pi]
        self.lower_bounds = [0.001, 2.0e9, 0.01]
        self.data_types = ['float', 'float', 'float']
        self.step_sizes = [None, 2e9, None]
        self.parameter_imprecisions = [1.0, 1.0, 0.1]
        self.parameter_units = [unit.rad, unit.Hz, unit.rad]
        self.parameter_locks = [False, False, False]
        self.parameter_names = ['depth', 'frequency', 'bias']
        self.parameter_symbols = [r"$x_m$", r"$x_f$", r"$x_{DC}$"]
        super().__init__(**kwargs)

        return

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0,
                  save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]

        depth = self.parameters[0]
        frequency = self.parameters[1]
        bias = self.parameters[2]

        transform = depth * (np.cos(2 * np.pi * frequency * propagator.t)) + bias

        if save_transforms:
            self.transform = (('t', transform, 'modulation'),)
        else:
            self.transform = None
        state1 = state/2.0 + state / 2.0 * np.exp(1j * transform)
        return [state1]


@register_node_types_all
class WaveShaper(SinglePath):
    """

    """

    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'WS'

        number_of_bins = 9
        self._number_of_bins = number_of_bins
        self.frequency_bin_width = 12e9
        self.extinction_ratio = 10 **( -35 / 10)

        #TODO: add test to make sure (at initialization that all these variables are the same length)
        # Then: also add one at runtime that ensure the .parameters variable is the same length
        self.number_of_parameters = 2 * number_of_bins

        self.default_parameters = [1.0] * number_of_bins + [0.1] * number_of_bins

        self.upper_bounds = [1.0] * number_of_bins + [2*np.pi] * number_of_bins
        self.lower_bounds = [self.extinction_ratio] * number_of_bins + [0.0] * number_of_bins
        self.data_types = 2 * number_of_bins * ['float']
        self.step_sizes = [None] * number_of_bins + [None] * number_of_bins
        self.parameter_imprecisions = [0.1] * number_of_bins + [0.1 * 2 * np.pi] * number_of_bins
        self.parameter_units = [None] * number_of_bins + [unit.rad] * number_of_bins
        self.parameter_locks = 2 * self.number_of_parameters * [False]
        self.parameter_names = ['amplitude{}'.format(ind) for ind in range(number_of_bins)] + \
                               ['phase{}'.format(ind) for ind in range(number_of_bins)]
        self.parameter_symbols = [r"$x_{a_{"+"{:+d}".format(ind)+r"}}$" for ind in range(-(number_of_bins-1)//2, (number_of_bins-1)//2+1)] + \
                                 [r"$x_{\phi_{"+"{:+d}".format(ind)+r"}}$" for ind in range(-(number_of_bins-1)//2, (number_of_bins-1)//2+1)]


        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]

        # Slice at into the first half (amp) and last half (phase)
        amplitudes = self.parameters[:self._number_of_bins]
        phases = self.parameters[self._number_of_bins:]

        n = np.floor(propagator.n_samples / ((1 / propagator.dt) / self.frequency_bin_width)).astype('int')
        N = np.shape(propagator.f)[0]
        tmp = np.ones((n, 1))

        a = np.array([i * tmp for i in amplitudes])
        p = np.array([i * tmp for i in phases])

        amp1 = np.concatenate(a)
        phase1 = np.concatenate(p)
    
        left = np.floor((propagator.n_samples - amp1.shape[0]) / 2).astype('int')
        right = propagator.n_samples - np.ceil((propagator.n_samples - amp1.shape[0]) / 2).astype('int')

        # we will pad amp1 and phase1 with zeros so they are the correct size
        pad_left = np.ones((left, 1)) * self.extinction_ratio
        pad_right = np.ones((N - right, 1)) * self.extinction_ratio

        # Concatenate the arrays together
        # We cannot use array assignment as it is not supported by autograd
        amplitude_mask = np.concatenate((pad_left, amp1, pad_right), axis=0)
        phase_mask = np.concatenate((pad_left, phase1, pad_right), axis=0)

        state = ifft_(ifft_shift_(amplitude_mask * np.exp(1j * phase_mask), ax=0) * fft_(state, propagator.dt), propagator.dt)

        if save_transforms:
            self.transform = (('f', amplitude_mask, 'amplitude'), ('f', phase_mask, 'phase'),)
        else:
            self.transform = None

        return [state]


@register_node_types_all
class DelayLine(SinglePath):
    """
    """
    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'DL'

        self.number_of_parameters = 8

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
        # [1e-12, 2e-12, 4e-12, 8e-12, 16e-12, 32e-12, 128e-12, 64e-12


        super().__init__(**kwargs)
        return

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]
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
                # field_short = (np.sqrt(1 - coupling_ratio) * field_short + 1j * np.sqrt(coupling_ratio) * field_long)
                # field_long = np.exp(1j * ifft_shift_(beta * length, ax=0)) * (
                #         1j * np.sqrt(coupling_ratio) * field_short_tmp + np.sqrt(1 - coupling_ratio) * field_long)
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

        return [ifft_(field_short, propagator.dt)]


#@register_node_types_all
class EDFA(SinglePath):
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
    def __init__(self, small_signal_gain=None, **kwargs):
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
        self.node_acronym = 'EDFA'

        self.number_of_parameters = 6
        self.upper_bounds = [50, 1612e-9, 1600-9, 1625e-9, 10, 10, 15, 1.5, 30]
        self.lower_bounds = [0, 1535e-9, 1530e-9, 1540e-9, 0, 0, 0, 0, 3]
        self.data_types = ['float'] * self.number_of_parameters
        self.step_sizes = [None] * self.number_of_parameters
        self.parameter_imprecisions = [1, 1e-9, 1e-9, 1e-9, 1e-3, 1e-3, 1, 0.1, 1] # placeholders, I don't really know
        self.parameter_units = [None, unit.m, unit.m, unit.m, unit.W, unit.W, None, None, None] # no dB unit exists in registry
        self.parameter_locks = [False] + [True] * (self.number_of_parameters - 1)
        self.parameter_names = ['max_small_signal_gain_dB', 'peak_wl', 'band_lower', 'band_upper', 'P_in_max', 'P_out_max', 'gain_flatness_dB', 'alpha', 'max_noise_fig_dB']
        self.default_parameters = [30, 1550e-9, 1520e-9, 1565e-9, 0.01, 0.1, 1.5, 1, 5]
        super().__init__(**kwargs)
        self._small_signal_gain = None
        self.noise_model = AdditiveNoise(noise_type='edfa ASE', noise_param=self)

        # save for noise propagation
        self._last_gain = None
        self._last_noise_factor = None

        #
        self.set_parameters_as_attr()

    def propagate(self, states, propagator, num_inputs=1, num_outputs=1, save_transforms=False):  # node propagate functions always take a list of propagators
        """
        """
        state = states[0]

        state_f = fft_(state, propagator.dt)

        gain = self._gain(state, propagator)

        self._noise_factor(state, propagator) # saves it for the noise call

        if save_transforms:
            self.transform = (('f', 10 * np.log10(ifft_shift_(np.sqrt(gain))), 'gain (dB)'),)

        return [ifft_(np.sqrt(gain) * state_f, propagator.dt)] # sqrt of gain, because gain is a ratio of power, not amplitude
    
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
        expected_power_dist = Planck * (ifft_shift_(propagator.f) + propagator.central_frequency) / 2 * \
            (self._last_noise_factor * self._last_gain - 1) * propagator.df

        return np.sqrt(expected_power_dist)


    def _gain(self, state, propagator):
        """
        Gain is defined as in [1], G = g / (1 + (g * P_in / P_max)^alpha), with g = small signal gain, G is true gain
        """
        if self._small_signal_gain is None:
            self._small_signal_gain = self._get_small_signal_gain(propagator)

        P_in = np.mean(power_(state)) # EDFAs saturation is affected by average power according to

        if P_in > self._P_in_max:
            raise ValueError(f'input signal {P_in} is greater than max input signal {self._P_in_max}')

        self._last_gain = self._small_signal_gain / (1 + (self._small_signal_gain * P_in / self._P_out_max)**self._alpha) 

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
        central_freq = speed_of_light / self._peak_wl
        lower_freq = speed_of_light / self._band_upper
        upper_freq = speed_of_light / self._band_lower
        d = np.maximum(central_freq - lower_freq, upper_freq - central_freq)

        beta = d**2 / (np.log(10**(self._gain_flatness_dB / 10)))
        g = 10**(self._max_small_signal_gain_dB / 10)
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
        wavelengths = [i*1e-9 for i in range(1520, 1565)]
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
    def __init__(self, **kwargs):
        self.node_lock = False
        self.node_acronym = 'PF'

        number_of_bases = 10
        self._number_of_bases = number_of_bases
        self.band_width = 5e12
        self.extinction_ratio = 10 ** (-35 / 10)

        self.number_of_parameters = 2 * number_of_bases

        self.default_parameters = [1.0] * number_of_bases + [0.0] * number_of_bases

        self.upper_bounds = [1.0] * number_of_bases + [2 * np.pi] * number_of_bases
        self.lower_bounds = [0.0] * number_of_bases + [0.0] * number_of_bases
        self.data_types = 2 * number_of_bases * ['float']
        self.step_sizes = [None] * number_of_bases + [None] * number_of_bases
        self.parameter_imprecisions = [0.1] * number_of_bases + [0.1 * 2 * np.pi] * number_of_bases
        self.parameter_units = [None] * number_of_bases + [unit.rad] * number_of_bases
        self.parameter_locks = 2 * self.number_of_parameters * [False]
        self.parameter_names = ['amplitude{}'.format(ind) for ind in range(number_of_bases)] + \
                               ['phase{}'.format(ind) for ind in range(number_of_bases)]
        self.parameter_symbols = [r"$x_{b_{" + "{:+d}".format(ind) + r"}}$" for ind in
                                  range(-(number_of_bases - 1) // 2, (number_of_bases - 1) // 2 + 1)] + \
                                 [r"$x_{p_{" + "{:+d}".format(ind) + r"}}$" for ind in
                                  range(-(number_of_bases - 1) // 2, (number_of_bases - 1) // 2 + 1)]

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

    # def get_waveform(self, t, coeffs):
    #     y = np.zeros_like(t)
    #     for m, coeff in enumerate(coeffs):
    #         y += coeff * np.cos(2 * np.pi * t * m + coeff)
    #     return y

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):  # node propagate functions always take a list of propagators
        state = states[0]

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

        # amplitude_mask = np.power(amplitude_mask, 4)
        # phase_mask = np.power(phase_mask, 2)/(2*np.pi)**2

        state = ifft_(ifft_shift_(amplitude_mask * np.exp(1j * phase_mask), ax=0) * fft_(state, propagator.dt),
                      propagator.dt)

        if save_transforms:
            self.transform = (('f', amplitude_mask, 'amplitude'), ('f', phase_mask, 'phase'),)
        else:
            self.transform = None
        return [state]