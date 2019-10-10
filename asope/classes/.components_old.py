
#%%
class AmplitudeModulator(Component):
    """
        Electro-optic amplitude modulator, driven by a single sine tone (for now).
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'amplitudemodulator'
        self.disp_name = 'Electro-Optic Amplitude Modulator'
        self.vpi = 1
        self.N_PARAMETERS = 3
        self.UPPER = [1, 24e9, 1]   # max shift, frequency, bias
        self.LOWER = [0, 6e9, 0]
        self.SIGMA = [0.1, 1e6, 0.1]
        self.MU = [0,0,0]
        self.DTYPE = ['float', 'float', 'float']
        self.DSCRTVAL = [None, 6e9, None]
        self.FINETUNE_SKIP = []
        self.splitter = False

    def simulate(self, env, field,  visualize=False):
        # extract attributes (parameters) of driving signal
        M = self.at[0]       # amplitude [V]
        NU = self.at[1]      # frequency [Hz]
        BIAS = self.at[2]     # voltage bias [V]
#        phase = (M)*(np.cos(2*np.pi* NU * env.t)) + (BIAS)

#        amp = np.power((M)*(np.cos(2*np.pi* NU * env.t)), 2)
#        amp = np.experiment(1j * np.pi / self.vpi * phase)

        amp = (M/2)*(np.cos(2*np.pi* NU * env.t)+1)

        # apply phase shift temporally
        field = field/2 * (1 + amp)

        if visualize:
            self.lines = (('t',amp, 'Amplitude Modulation'),)

        return field

    def newattribute(self):

        at = []
        for i in range(self.N_PARAMETERS):
            at.append(self.randomattribute(self.LOWER[i], self.UPPER[i], self.DTYPE[i], self.DSCRTVAL[i]))
        self.at = at
        return at

    def mutate(self):
        mut_loc = np.random.randint(0, self.N_PARAMETERS)
        self.at[mut_loc] = self.randomattribute(self.LOWER[mut_loc], self.UPPER[mut_loc],        self.DTYPE[mut_loc], self.DSCRTVAL[mut_loc])
        return self.at

# ----------------------------------------------------------
class AWG(Component):
    """
        Simple AWG in the temporal domain to apply phase shifts on the pulses
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'awg'
        self.disp_name = 'AWG Phase Modulator'
        self.N_PARAMETERS = 2
        self.UPPER = [8] + [np.pi]   # number of steps + 1, phase at each step
        self.LOWER = [1] + [-np.pi]
        self.DTYPE = ['int', 'float']
        self.DSCRTVAL = [1, None]
        self.FINETUNE_SKIP = [0] #index to skip when fine-tuning using gradient descent
        self.splitter = False

    def simulate(self, env, field, visualize=False):
        # extract attributes, first index is the number of steps - which affects the other attributes
        nlevels = self.at[0] + 1

        # phase to put on each step
        phasevalues = [0] + self.at[1:]

        # create step pattern, with steps centered where the original pulses are
        # (there are likely better ways to approach this, but how? without having many parameters)
        timeblock = np.round(1/env.dt/env.f_rep).astype('int')
        tmp = np.ones(timeblock)
        oneperiod = np.array([]).astype('float')
        for i in range(0,nlevels):
            oneperiod = np.concatenate((oneperiod, tmp*phasevalues[i]))

        # tile/repeat the step-waveform for the whole simulation window
        phasetmp = np.tile(oneperiod, np.ceil(env.n_samples/len(oneperiod)).astype('int'))

        shift1 = timeblock//2
        phasetmp = phasetmp[shift1:]
        phase = phasetmp[0:env.n_samples].reshape(env.n_samples, 1)

        # apply phase profile in the temporal domain, and a type of loss can be added to reduce number of steps
        field = field * np.exp(1j * phase)


        if visualize:
            self.lines = (('t',phase, 'Arbitrary Phase Pattern'),)

        return field

    def newattribute(self):
        # carefully create new attributes, as you must consider the number of steps which changes the length of the attribute (parameter) list
        n = self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])
        vals = []
        for i in range(n):
            vals.append(self.randomattribute(self.LOWER[1], self.UPPER[1], self.DTYPE[1], self.DSCRTVAL[1]))
        at = [n] + vals
        self.at = at
        return at


    def mutate(self):
        # also must be careful to mutate a list of attributes
        at = self.at
        mut_loc = np.random.randint(0, len(at))
        if mut_loc == 0: # mutates the number of steps
            n_mut = self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])
            if n_mut > at[0]: # mutated to have more steps than before
                new_vals = []
                for i in range(n_mut-at[0]):
                    new_vals.append(self.randomattribute(self.LOWER[1], self.UPPER[1], self.DTYPE[1], self.DSCRTVAL[1]))
                vals = at[1:] + new_vals
                at = [n_mut] + vals
            else:
                vals = at[1:n_mut+1]
                at = [n_mut] + vals
        else: # keep the same number of steps, but change the values at each step
            at[mut_loc] = self.randomattribute(self.LOWER[1], self.UPPER[1], self.DTYPE[1], self.DSCRTVAL[1])
        self.at = at
        return at















import autograd.numpy as np
from itertools import count
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

from classes.environment import OpticalField, OpticalField_CW, OpticalField_Pulse

"""
ASOPE
|- components.py

Each component in the simulations is described by a custom 'Component' class. Within this class various physical parameters are stored (for example dispersion of a fiber) which are neeed for the simulation of an experimental setup containing that component, but also variables for the optimization. 

Each component class a number of important class variables:
    - type = what type of component is this (awg, powersplitter, fiber, etc)
    - N_PARAMETERS = how many parameters on the component are to be optimized (as a list)
    - UPPER = upper bound on your optimization parameters (as a list of length N_PARAMETERS)
    - LOWER = lower bound on your optimization parameters (as a list of length N_PARAMETERS)
    - DTYPE = the datatype of each parameter you are optimizing, either int or float (as a list)
    - DSCRTVAL = the discretization step (resolution) when generating a random attribute for the parameters (None if you want continuous)
    - FINETUNE_SKIP = this is for the fine-tuning using gradient descent. As some parameters are integers (for example, the number of steps on the AWG) and the grad-desc cannot deal with ints, we skip is. This is a list of the indices to skip
    - splitter = Defines whether this component will have more than one output or input (only certain component types support multiple input/outputs)

There is also important class functions:
    - datasheet() = contains all the information about the component
    - simulate() = simulates the transformation of the component to the input
    - mutate() = in the GA, used to mutate the attributes on each component
    - newattribute() = will create random attribute for ALL parameters of the component

    - randomattribute() = based on the settings for the component (bounds, discretization, etc), will generate ONE random attribute (setting) for the component
"""


class Component(object):
	def __init__(self):
		"""
			Initialize each component, and saves the datasheet to as class variables
		"""
		self.datasheet()
		self.updateinstances()
		self.name = str(self.type) + str(self.id)

	def updateinstances(self):
		"""
			Keeps track of how many of each component is in the setup to ensure they are distinguishable (fiber0, fiber1, etc)
		"""
		self.id = next(self._num_instances)

	def resetinstances(self):
		self._num_instances = count(0)

	def datasheet(self):
		"""
			Different for each component, but saves all important parameters for the physical device
		"""
		return

	def simulate(self):
		"""
			Simulates the transformation of the component on the pulse
		"""
		raise ValueError('Not implemented yet')

	def newattribute(self):
		"""
			Creates a list of attributes (parameters) for the device
		"""
		at = []
		for i in range(self.N_PARAMETERS):
			at.append(self.randomattribute(self.LOWER[i], self.UPPER[i], self.DTYPE[i], self.DSCRTVAL[i]))
		self.at = at
		return at

	def mutate(self):
		"""
			Mutates the list of attributes (parameters) for the device, used in the GA
		"""
		mut_loc = np.random.randint(0, self.N_PARAMETERS)
		self.at[mut_loc] = self.randomattribute(self.LOWER[mut_loc], self.UPPER[mut_loc], self.DTYPE[mut_loc],
		                                        self.DSCRTVAL[mut_loc])
		return self.at


#%%
class PhotonicAWG(Component):
    """

    """
    _num_instances = count(0)

    def datasheet(self):
        self.type = 'photonic awg'
        self.disp_name = 'Photonic Driven AWG'

        self.phasemodulator = PhaseModulator()
        self.phasemodulator.N_PARAMETERS = 1
        self.phasemodulator.UPPER.pop()
        self.phasemodulator.LOWER.pop()
        self.phasemodulator.SIGMA.pop()
        self.phasemodulator.MU.pop()
        self.phasemodulator.DTYPE.pop()
        self.phasemodulator.DSCRTVAL.pop()

        self.waveshaper = WaveShaper()
        self.cw = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)

        self.N_PARAMETERS = self.phasemodulator.N_PARAMETERS + self.waveshaper.N_PARAMETERS + 1

        self.UPPER = self.phasemodulator.UPPER + self.waveshaper.UPPER + [100]
        self.LOWER = self.phasemodulator.LOWER + self.waveshaper.LOWER + [10]
        self.DTYPE = self.phasemodulator.DTYPE + self.waveshaper.DTYPE + ['float']
        self.DSCRTVAL = self.phasemodulator.DSCRTVAL + self.waveshaper.DSCRTVAL + [None]
        self.SIGMA = self.phasemodulator.SIGMA + self.waveshaper.SIGMA + [0.01]
        self.MU = self.phasemodulator.MU + self.waveshaper.MU + [0.0]
        self.FINETUNE_SKIP = []
        self.splitter = False
        self.AT_NAME = self.phasemodulator.AT_NAME + self.waveshaper.AT_NAME + ['awg power']




    def simulate(self, env, field, visualize=False):
        # attribute list is extracted

        # extract the attributes to the two building block components
        at_phasemodulator = self.at[0:self.phasemodulator.N_PARAMETERS]
        at_waveshaper = self.at[self.phasemodulator.N_PARAMETERS:-1]
        at_power = self.at[-1]

        # set the parameters to the inner components
        self.phasemodulator.at = at_phasemodulator
        self.waveshaper.at = at_waveshaper

        # simulate these components
        awg_optical = self.cw.field
        awg_optical = self.phasemodulator.simulate(self.cw, awg_optical)
        awg_optical = self.waveshaper.simulate(self.cw, awg_optical)
        awg_electric = P(awg_optical)

        def extract_matrix_from_continuous(env, field):
            matrix_elements = FFT(field, env.dt)[env.inds].flatten()
            matrix = np.zeros([env.dim, env.dim]).astype('complex')
            for ii in range(0, env.dim, 1):
                matrix[ii, :] = matrix_elements[env.dim - ii - 1: 2 * (env.dim - 1) + 1 - ii]
            return matrix / np.max(np.abs(matrix)) / np.sqrt(env.dim)

        field = field * np.exp(1j * at_power * awg_electric)

        env.mat = np.matmul(extract_matrix_from_continuous(env, field), env.mat)




        if visualize:
            self.lines = (('t', awg_electric, 'Photonic AWG Modulation'),)

        return field

#%%
class QuantumPhaseGate(Component):
    """
        Waveshaper, with amplitude and phase masks. Currently, the mask profile are made with a polynomial (parameters are the polynomial coefficients) and then clipped to valid levels (0-1 for amplitude, 0-2pi for phase). This is admittedly likely not the best solution - but for now it can work.
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'waveshaper'
        self.disp_name = 'Programmable Filter - Phase Gate'
        self.bitdepth = 8
        self.res = 12e9     # resolution of the waveshaper
        self.n_windows = 3
        self.N_PARAMETERS = 2*self.n_windows
        self.UPPER = self.n_windows*[1.0] + self.n_windows*[2*np.pi]
        self.LOWER = self.n_windows*[0.0] + self.n_windows*[0.0]
        self.SIGMA = self.n_windows*[0.05] + self.n_windows*[0.5*np.pi]
        self.MU = self.n_windows*[0.0] + self.n_windows*[0.0]
        self.DTYPE = self.n_windows * ['float'] + self.n_windows * ['float']

        self.DSCRTVAL = self.N_PARAMETERS * [None]
        # self.DSCRTVAL = self.n_windows * [1/(2**self.bitdepth-1)] + self.n_windows * [2*np.pi/(2**self.bitdepth-1) ]
        self.FINETUNE_SKIP = []
        self.splitter = False

        self.AT_NAME = ['window amplitude {}'.format(j - int(np.floor(self.n_windows/2))) for j in range(0, self.n_windows, 1)] + ['window phase {}'.format(j - int(np.floor(self.n_windows/2))) for j in range(0, self.n_windows, 1)]


    def temp_func(self, left, right, i, vals):
        if i >= left and i < right:
            return vals[i-left][0]
        else:
            return 0

    def simulate(self, env, field, visualize = False):
        # Slice at into the first half (amp) and last half (phase)
        ampvalues = self.at[0:self.n_windows]
        phasevalues = self.at[self.n_windows:]

        n = np.floor(env.n_samples/((1/env.dt)/self.res)).astype('int')
        N = np.shape(env.f)[0]
        tmp = np.ones((n,1))

        a = np.array([i*tmp for i in ampvalues])
        p = np.array([i*tmp for i in phasevalues])

        amp1 = np.concatenate(a)
        phase1 = np.concatenate(p)

        left = np.floor((env.n_samples - amp1.shape[0])/2).astype('int')
        right = env.n_samples - np.ceil((env.n_samples - amp1.shape[0])/2).astype('int')

        # we will pad amp1 and phase1 with zeros so they are the correct size
        padleft = np.zeros((left, 1))
        padright = np.zeros((N-right, 1))

        # Concatenate the arrays together
        # We cannot use array assignment as it is not supported by autograd
        ampmask = np.concatenate((padleft, amp1, padright), axis=0)
        phasemask = np.concatenate((padleft, phase1, padright), axis=0)

        Af = ampmask * np.exp(1j*(phasemask)) * FFT(field, env.dt)
        field = IFFT( Af, env.dt )

        env.mat = np.matmul( np.diag( np.array(ampvalues) * np.exp(1j * np.array(phasevalues))), env.mat )
        env.mat = env.mat / np.max(np.abs(env.mat))

        if visualize:
            self.lines = (('f',ampmask,'WaveShaper Amplitude Mask'),('f', phasemask,'WaveShaper Phase Mask'),)
        return field

    def newattribute(self):
        at = []
        for i in range(self.N_PARAMETERS):
            at.append(self.randomattribute(self.LOWER[i], self.UPPER[i], self.DTYPE[i], self.DSCRTVAL[i]))
        self.at = at
        return at


    def mutate(self):
        mut_loc = np.random.randint(0, self.N_PARAMETERS)
        self.at[mut_loc] = self.randomattribute(self.LOWER[mut_loc], self.UPPER[mut_loc],        self.DTYPE[mut_loc], self.DSCRTVAL[mut_loc])
        return self.at