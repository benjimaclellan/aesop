import autograd.numpy as np
from itertools import count
import autograd.scipy as sp
from classes.components import WaveShaper, PhaseModulator
from classes.environment import OpticalField_CW

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

	def randomattribute(self, low=0.0, high=1.0, dtypes='float', dscrtval=None):
		"""
			Common function for creating a new random attribute, based on the bounds (upper and lower), datatype, and discretization
		"""

		if dtypes == 'float':
			if dscrtval is not None:
				at = round(np.random.uniform(low, high) / dscrtval) * dscrtval
			else:
				at = np.random.uniform(low, high)

		elif dtypes == 'int':
			if dscrtval is not None:
				at = np.round(np.random.randint(low / dscrtval, high / dscrtval)) * dscrtval
			else:
				at = np.random.randint(low, high)
		else:
			raise ValueError('Unknown datatype when making a new attribute')

		return at

#%%
class QuantumPhaseModulator(Component):
	"""
		Electro-optic phase modulator, driven by a single sine tone (for now).
	"""
	_num_instances = count(0)
	def datasheet(self):
		self.type = 'phasemodulator'
		self.disp_name = 'Electro-Optic Phase Modulator'
		self.vpi = 1
		self.N_PARAMETERS = 2
		self.UPPER = [4, 2*np.pi]   # max shift, offset
		self.LOWER = [0, 0]
		self.SIGMA = [0.05, 0.01]
		self.MU = [0.0, 0.0]
		self.DTYPE = ['float', 'float']
		self.DSCRTVAL = [None, None]
		self.FINETUNE_SKIP = []
		self.splitter = False
		self.AT_NAME = ['mod depth', 'mod offset']


		dim = 2
		x, y = np.arange(0, dim) - dim//2 + 1, np.arange(0, dim) - dim//2 + 1
		X, Y = np.meshgrid(x, y)
		self.IJ = X-Y


	def simulate(self, env, mat,  visualize=False):
		# extract attributes (parameters) of driving signal
		M = self.at[0]      # phase amplitude [V/Vpi]
		NU = 12e9           # frequency [Hz]
		SHIFT = self.at[1]  # phase offset

		U = sp.special.jn(self.IJ, M)
		mat = np.matmul(U, mat)


		if visualize:
			self.lines = ((None, None, None),)

		return mat

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


#%%
class PhotonicAWG(Component):
	_num_instances = count(0)

	def datasheet(self):
		self.cw = OpticalField_CW(n_samples=2**14, window_t=10e-9, peak_power=1)
		self.phasemodulator = PhaseModulator()
		self.waveshaper = WaveShaper()


		self.type = 'photonicawg'
		self.disp_name = 'Photonic AWG'
		self.N_PARAMETERS = self.phasemodulator.N_PARAMETERS + self.waveshaper.N_PARAMETERS
		self.UPPER = self.phasemodulator.UPPER + self.waveshaper.UPPER
		self.LOWER = self.phasemodulator.LOWER + self.waveshaper.LOWER
		self.SIGMA = self.phasemodulator.SIGMA + self.waveshaper.SIGMA
		self.MU = self.phasemodulator.MU + self.waveshaper.MU
		self.DTYPE = self.phasemodulator.DTYPE + self.waveshaper.DTYPE
		self.DSCRTVAL = self.phasemodulator.DSCRTVAL + self.waveshaper.DSCRTVAL
		self.FINETUNE_SKIP = []
		self.splitter = False
		self.AT_NAME = self.phasemodulator.AT_NAME + self.waveshaper.AT_NAME

		self.dim = 2
		self.fsr = 12e9
		x, y = np.arange(0, self.dim) - self.dim // 2 + 1, np.arange(0, self.dim) - self.dim // 2 + 1
		X, Y = np.meshgrid(x, y)
		self.IJ = X - Y

		return

	def simulate(self, env, mat, visualize=False):
		at_phasemodulator = self.at[0:self.phasemodulator.N_PARAMETERS]
		at_waveshaper = self.at[self.phasemodulator.N_PARAMETERS+1:]

		At = self.cw.At
		At = self.phasemodulator.simulate(self.cw, At)
		At = self.waveshaper.simulate(self.cw, At)

		PAt = np.power(np.abs(At), 2)

		freqs = np.arange(0, self.waveshaper.N_PARAMETERS-1) * self.fsr
		coefficients = self.DFT(PAt, freq, self.cw.t)



		if visualize:
			self.lines = ((None, None, None),)
		return mat

	def DFT(self, waveform, frequencies, time):
	    N = len(waveform)
	    coefficients = np.zeros_like(frequencies).astype('complex')
	    for idx, frequency in enumerate(frequencies):
		    coefficients[idx] = np.sum(waveform * np.exp(-1j * 2 * np.pi * frequency * time)) / N
	    return coefficients



#%%
class QuantumPhaseGate(Component):
	"""
	"""
	_num_instances = count(0)
	def datasheet(self):
		self.type = 'waveshaper'
		self.disp_name = 'Programmable Filter - Phase Gate'
		self.bitdepth = 8
		self.res = 12e9     # resolution of the waveshaper
		self.n_windows = 2
		self.N_PARAMETERS = self.n_windows
		self.UPPER = self.n_windows * [2 * np.pi]
		self.LOWER = self.n_windows * [0.0]
		self.SIGMA = self.n_windows * [0.5 * np.pi]
		self.MU = self.n_windows * [0.0]
		self.DTYPE = self.n_windows * ['float']
		# self.N_PARAMETERS = 2*self.n_windows
		# self.UPPER = self.n_windows*[1.0] + self.n_windows*[2*np.pi]
		# self.LOWER = self.n_windows*[0.0] + self.n_windows*[0.0]
		# self.SIGMA = self.n_windows*[0.05] + self.n_windows*[0.5*np.pi]
		# self.MU = self.n_windows*[0.0] + self.n_windows*[0.0]
		# self.DTYPE = self.n_windows * ['float'] + self.n_windows * ['float']

		self.DSCRTVAL = self.N_PARAMETERS * [None]
		self.FINETUNE_SKIP = []
		self.splitter = False

		self.AT_NAME = ['phase {}'.format(j - int(np.floor(self.n_windows/2))) for j in range(0, self.n_windows, 1)]

	def simulate(self, env, mat, visualize=False):
		phasevalues = self.at
		U = np.diag(np.exp(1j * np.array(phasevalues)))

		mat = np.matmul(U, mat)

		if visualize:
			self.lines = (None,)
		return mat

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