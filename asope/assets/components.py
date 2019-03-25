import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from copy import copy, deepcopy
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
import inspect

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
        raise ValueError('Not implemented yet') 
    
    def mutate(self):
        """
            Mutates the list of attributes (parameters) for the device, used in the GA
        """
        raise ValueError('Not implemented yet') 
        
        
    def randomattribute(self, low=0.0, high=1.0, dtypes='float', dscrtval=None):
        """
            Common function for creating a new random attribute, based on the bounds (upper and lower), datatype, and discretization
        """
        
        if dtypes == 'float':
            if dscrtval is not None:  
                at = round(np.random.uniform( low, high )/dscrtval) * dscrtval 
            else:
                at = np.random.uniform( low, high )
                
        elif dtypes == 'int':
            if dscrtval is not None:    
                at = np.round(np.random.randint( low/dscrtval, high/dscrtval))*dscrtval
            else: 
                at = np.random.randint(low, high)
        else:
            raise ValueError('Unknown datatype when making a new attribute')
    
        return at

# ----------------------------------------------------------
# Here we now implement each component - and more can be added easily or adapted to a different purpose (ie quantum). It is also trivial to change how the device is simulated without changing the rest of the code, provided the general format is followed
# ----------------------------------------------------------

class Fiber(Component):
    """
        Simple dispersive fiber. Only considers second order dispersion for now.
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'fiber'
        self.beta = [2e-20]     # second order dispersi on (SI units)
        self.N_PARAMETERS = 1
        self.UPPER = [2000]
        self.LOWER = [0]
        self.DTYPE = ['int']
        self.DSCRTVAL = [1]
        self.FINETUNE_SKIP = None
        self.splitter = False
        
    def simulate(self, env, At, visualize=False):
        
        # attribute list is extracted. For fiber, only one parameter which is length
        fiber_len = self.at[0]   
        
        # calculate the dispersion operator in the spectral domain
        D = np.zeros(env.f.shape).astype('complex')
        for n in range(0, len(self.beta)):    
            D += self.beta[n] * np.power(2*np.pi*env.f, n+2) / np.math.factorial(n+2)
        D = -1j * D
        
        # apply dispersion
        Af = np.exp(fiber_len * D) * FFT(At, env.dt)
        At = IFFT( Af, env.dt )
        
        # this visualization functionality was broken, may be fixed later
#        if visualize:
#            self.lines = ((None),)
        return At
    
    def newattribute(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at
    
    def mutate(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at
        
# ----------------------------------------------------------
class PhaseModulator(Component):
    """
        Electro-optic phase modulator, driven by a single sine tone (for now).
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'phasemodulator'
        self.vpi = 1
        self.N_PARAMETERS = 3
        self.UPPER = [1, 200e6, 1]   # max shift, frequency, bias
        self.LOWER = [0, 1e6, 0]
        self.DTYPE = ['float', 'float', 'float']
        self.DSCRTVAL = [None, 1e6, None]
        self.FINETUNE_SKIP = False
        self.splitter = False
    
    def simulate(self, env, At,  visualize=False):  
        # extract attributes (parameters) of driving signal
        M = self.at[0]       # amplitude [V]
        NU = self.at[1]      # frequency [Hz]
        BIAS = self.at[2]     # voltage bias [V]
        phase = (M * np.pi / 2 / self.vpi)*(np.cos(2*np.pi* NU * env.t)) + (BIAS * np.pi / 2 / self.vpi)
        
        # apply phase shift temporally
        At = At * np.exp(1j * phase)                
        
#        if visualize:
#            self.lines = (('t',phase),)
        
        return At

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
        self.N_PARAMETERS = 2
        self.UPPER = [8] + [np.pi]   # number of steps + 1, phase at each step
        self.LOWER = [1] + [-np.pi]
        self.DTYPE = ['int', 'float']
        self.DSCRTVAL = [1, None]
        self.FINETUNE_SKIP = [0] #index to skip when fine-tuning using gradient descent
        self.splitter = False
    
    def simulate(self, env, At, visualize=False):   
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
        phasetmp = np.tile(oneperiod, np.ceil(len(env.t)/len(oneperiod)).astype('int') )
        shift1 = timeblock//2
        phasetmp = phasetmp[shift1:]
        phase = phasetmp[0:len(env.t)] 
        
        # apply phase profile in the temporal domain, and a type of loss can be added to reduce number of steps
        At = At * np.exp(1j * phase) * (.97**nlevels) #loss here
        
        return At
    
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

# ----------------------------------------------------------
class WaveShaper(Component):
    """
        Waveshaper, with amplitude and phase masks. Currently, the mask profile are made with a polynomial (parameters are the polynomial coefficients) and then clipped to valid levels (0-1 for amplitude, 0-2pi for phase). This is admittedly likely not the best solution - but for now it can work.
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'waveshaper'
        self.res = 12e9     # resolution of the waveshaper
        self.N_PARAMETERS = 4 * 2 #using a 4-th order polynomial, for now
        self.UPPER = [1,1,1,2] + [1,1,1,2]  #bounds of polynomial coefficients
        self.LOWER = [-1,-1,-1,0] + [-1,-1,-1,0]
        self.DTYPE = self.N_PARAMETERS * ['float']
        self.DSCRTVAL = self.N_PARAMETERS * [None]
        self.FINETUNE_SKIP = None
        self.splitter = False
    
    def simulate(self, env, At, visualize = False):
        # extract phase and amplitude mask values (polynomial coefficients)
        ampvalues = self.at[0:self.N_PARAMETERS//2]
        phasevalues = self.at[self.N_PARAMETERS//2:]
        
        # create the amplitude mask polynomial and clip between 0 and 1
        amp = np.polyval(ampvalues, env.f/np.max(env.f))
        amp[amp > 1] = 1
        amp[amp < 0] = 0
        
        # create the phase mask polynomial and clip between 0 and 2pi
        phase = np.polyval(phasevalues, env.f/np.max(env.f))
        phase[phase > 2*np.pi] = 2*np.pi
        phase[phase < 0] = 0
        
        # discretize based on the resolution of the waveshaper
        n = int( np.floor( self.res/env.df ) )
        ampmask = self.windowmask(amp, env.N, n)
        phasemask = self.windowmask(phase, env.N, n)
        
        # apply the waveshaper in the spectral domain and FFT back to temporal
        Af = ampmask * np.exp(1j * phasemask) * FFT(At, env.dt)
        At = IFFT( Af, env.dt )
        
        return At
    
    def windowmask(self, mask, N, n):
        # helper function for discretizing the masks based on the resolution
        
        totalbins = int(np.ceil( N / n ) )   # number of bins that fit into the window
        center = N // 2     # center of the window
        bins2center = int( np.floor( center / n ) )
        offset = abs((bins2center*n) - center) - n//2
        coursemask = np.zeros(mask.shape)
        slc = (0, n-offset)
        coursemask[ slc[0]:slc[1]] = np.mean(mask[ slc[0]:slc[1]])
        for i in range(0,totalbins):
            slc = ( offset + i*n, offset + (i+1)*n)
            coursemask[ slc[0]:slc[1]] = np.mean(mask[ slc[0]:slc[1]])
        slc = (offset + (totalbins)*n,-1) 
        coursemask[ slc[0]:slc[1]] = np.mean(mask[ slc[0]:slc[1]])
        return coursemask
        
        
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
class PowerSplitter(Component):
    """
        Simple balanced (3dB for two arms) power splitter. No parameters to optimize.
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'powersplitter'
        self.N_PARAMETERS = 0
        self.UPPER = []
        self.LOWER = []
        self.DTYPE = ['float']
        self.DSCRTVAL = [None]
        self.FINETUNE_SKIP = 0
        self.splitter = True
        
    def simulate(self, env, At_in, num_outputs, visualize=False):        
        # ensure there is maximum 2 inputs/outputs (for now)

        # this is a hacky fix -- look into a better solution later        
        if At_in.ndim == 1:
            At_in = np.atleast_2d(At_in).T

        
        num_inputs = At_in.shape[1]
        assert num_inputs <= 2
        assert num_outputs <= 2
        
        # this is kinda overkill, but can be extended to multi-path powersplitters (ie tritters) if wanted
        XX,YY = np.meshgrid(np.linspace(0,num_outputs-1, num_outputs), np.linspace(0,num_inputs-1, num_inputs))
        
        # in the case of 2x2 splitter, this works, but should check for more arms
        S = np.sqrt(1/2) * np.exp(np.abs(XX - YY) * 1j * np.pi )
        
        
#        S = np.sqrt(1/2) * np.array([[1,1],[1,1*np.exp(1j*np.pi)]])
#        if num_inputs == 1:
##            S = S[:,0]
#            S = S[0,:].reshape([1,2])
#        if num_outputs == 1:
##            S = S[0,:]
#            S = S[:,0].reshape([2,1])
                

        # apply scattering matrix to inputs and return the outputs        
        At_out = At_in.dot(S)
        
        return At_out
    
    def newattribute(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at
    
    def mutate(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at



# ----------------------------------------------------------
class FrequencySplitter(Component):
    """
        Frequency splitter for splitting spectrum into two spatial paths. Currently using one paramter (attribute), which sets where the (even) split occurs. However, it can trivially be extended to more complex selection of wavelengths
    """
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'frequencysplitter'
        self.N_PARAMETERS = 1
        self.UPPER = [0.06]
        self.LOWER = [-0.06]
        self.DTYPE = ['float']
        self.DSCRTVAL = [0.02]
        self.FINETUNE_SKIP = 0
        self.splitter = True
        
    def simulate(self, env, At_in, num_outputs, visualize=False):
        # ensuring that, for now, we only have maximum ONE input. Please use a powersplitter for coupling (easier to deal with)
        num_inputs = At_in.shape[1]
        assert num_inputs <= 1
        
        # ensuring that, for now, we only have maximum two outputs
        assert num_outputs <= 2

        # collect the input (single input path)
        Af_in = FFT(At_in[:,0], env.dt)
                
        # extract the frequency location to split at (can be extended to have two)
        splits = (env.f[0]-env.df, self.at[0]*env.f[-1])
        split1 = min(splits)
        split2 = max(splits)
        
        # create masks, which are used to select the frequencies on each outgoing spatial path
        mask1 = np.ones(env.N)
        mask1[env.f <= split1] = 0; mask1[env.f > split2] = 0   
        
        # second mask is the NOT of the first
        mask2 = np.logical_not(mask1).astype('float')
        
        # apply masks and stack the two outputs to be saved and sent to next components
        At_out = np.stack((IFFT(mask1*Af_in, env.dt), IFFT(mask2*Af_in,env.dt)), axis=1)
        return At_out
    
    def newattribute(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at
    
    def mutate(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at


