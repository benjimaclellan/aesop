import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from copy import copy, deepcopy
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

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
        self.datasheet()
        self.updateinstances()
        self.name = str(self.type) + str(self.id)
        
    def updateinstances(self):
        self.id = next(self._num_instances)
        
    def datasheet(self):
        return
    
    def simulate(self):
        raise ValueError('Not implemented yet')  
    
    def newattribute(self):
        raise ValueError('Not implemented yet') 
    
    def mutate(self):
        raise ValueError('Not implemented yet') 
        
    def randomattribute(self, low=0.0, high=1.0, dtypes='float', dscrtval=None):
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
"""

"""
class Fiber(Component):
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'fiber'
        self.beta = [2e-20]
        self.Tr = 0.5
        self.gamma = 100
        self.N_PARAMETERS = 1
        self.UPPER = [2000]
        self.LOWER = [0]
        self.DTYPE = ['int']
        self.DSCRTVAL = [1]
        self.FINETUNE_SKIP = None
        self.splitter = False
        
    def simulate(self, env, At, visualize=False):
        
        fiber_len = self.at[0]   
        
        # calculate the dispersion operator in the spectral domain
        D = np.zeros(env.f.shape).astype('complex')
        for n in range(0, len(self.beta)):    
            D += self.beta[n] * np.power(2*np.pi*env.f, n+2) / np.math.factorial(n+2)
        D = -1j * D
        
        # apply dispersion
        Af = np.exp(fiber_len * D) * env.FFT(At, env.dt)
        At = env.IFFT( Af, env.dt )
        
        if visualize:
            self.lines = ((None),)
        
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
"""

"""
class PhaseModulator(Component):
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'phasemodulator'
        self.vpi = 1
        self.N_PARAMETERS = 3
        self.UPPER = [10, 200e6, 2*np.pi]   # max shift, frequency, phase offset
        self.LOWER = [0, 1e6, 0]
        self.DTYPE = ['float', 'float', 'float']
        self.DSCRTVAL = [None, 1e6, None]
        self.FINETUNE_SKIP = False
        self.splitter = False
    
    def simulate(self, env, At,  visualize=False):
#        env = env_in[0]
        
        M = self.at[0]       # amplitude []
        NU = self.at[1]      # frequency [Hz]
        PHI = self.at[2]     # phase offset [rad]
        phase = (M/2)*np.cos(2*np.pi* NU * env.t + PHI) + M/2
        
        ## Apply phase modulator in time-domain, and update frequency
#        env.At = env.At * np.exp(1j * phase)                
#        env.Af = env.FFT(env.At, env.dt)
        
        At = At * np.exp(1j * phase)                
#        env.Af = env.FFT(env.At, env.dt)
        
        if visualize:
            self.lines = (('t',phase),)
        
        return At #[env]

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
"""

"""
class AWG(Component):
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'awg'
        self.N_PARAMETERS = 2
        self.UPPER = [8] + [np.pi]   # max shift, frequency, phase offset
        self.LOWER = [1] + [-np.pi]
        self.DTYPE = ['int', 'float']
        self.DSCRTVAL = [1, None]
        self.FINETUNE_SKIP = [0] #index to skip when fine-tuning using gradient descent
        self.splitter = False
    
    def simulate(self, env, At, visualize=False):        
        nlevels = self.at[0] + 1
        phasevalues = [0] + self.at[1:]
        
        timeblock = np.round(1/env.dt/env.f_rep).astype('int')
        tmp = np.ones(timeblock)
        oneperiod = np.array([]).astype('float')
        
        for i in range(0,nlevels):
            oneperiod = np.concatenate((oneperiod, tmp*phasevalues[i]))            
        phasetmp = np.tile(oneperiod, np.ceil(len(env.t)/len(oneperiod)).astype('int') )

        shift1 = timeblock//2
        
        phasetmp = phasetmp[shift1:]
        phase = phasetmp[0:len(env.t)] 
        
        At = At * np.exp(1j * phase) #* (.99**nlevels) #loss here
        
#        if visualize:
#            self.lines = (('t',phase),)
        
        return At
    
    def newattribute(self):
        n = self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])
        vals = []
        for i in range(n):
            vals.append(self.randomattribute(self.LOWER[1], self.UPPER[1], self.DTYPE[1], self.DSCRTVAL[1]))
        at = [n] + vals
        self.at = at
        return at
    
    
    def mutate(self):
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
        else:
            at[mut_loc] = self.randomattribute(self.LOWER[1], self.UPPER[1], self.DTYPE[1], self.DSCRTVAL[1])
        self.at = at
        return at


"""

"""
class WaveShaper(Component):
    _num_instances = count(0)
    def datasheet(self):
        self.type = 'waveshaper'
        self.res = 12e9
        self.N_PARAMETERS = 4 * 2
        self.UPPER = [1,1,1,2] + [1,1,1,2]
        self.LOWER = [-1,-1,-1,0] + [-1,-1,-1,0]
        self.DTYPE = self.N_PARAMETERS * ['float']
        self.DSCRTVAL = self.N_PARAMETERS * [None]
        self.FINETUNE_SKIP = None
        self.splitter = False
    
    def simulate(self, env, At, visualize = False):
        
        ampvalues = self.at[0:self.N_PARAMETERS//2]
        phasevalues = self.at[self.N_PARAMETERS//2:]
        
        amp = np.polyval(ampvalues, env.f/np.max(env.f))
        amp[amp > 1] = 1
        amp[amp < 0] = 0
        
        phase = np.polyval(phasevalues, env.f/np.max(env.f))
        phase[phase > 2*np.pi] = 2*np.pi
        phase[phase < 0] = 0
        
        
        n = int( np.floor( self.res/env.df ) )
        ampmask = self.windowmask(amp, env.N, n)
        phasemask = self.windowmask(phase, env.N, n)
        
        Af = ampmask * np.exp(1j * phasemask) * env.FFT(At, env.dt)
        At = env.IFFT( Af, env.dt )
        
#        if visualize:
#            self.lines = (('f',ampmask), ('f',phasemask))
        
        return At
    
    def windowmask(self, mask, N, n):
        totalbins = int(np.ceil( N / n ) )
        center = N // 2
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





class PowerSplitter(Component):
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
        
        num_inputs = At_in.shape[1]
        
        assert num_inputs <= 2
        assert num_outputs <= 2
        
        XX,YY = np.meshgrid(np.linspace(0,num_outputs-1, num_outputs), np.linspace(0,num_inputs-1, num_inputs))
        S = np.sqrt(1/num_outputs) * np.exp(np.abs(XX - YY) * 1j * np.pi / num_outputs)

        At_out = At_in.dot(S)
   
#        if visualize:
#            self.lines = ((None),)    
        return At_out
    
    def newattribute(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at
    
    def mutate(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at




class FrequencySplitter(Component):
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
        num_inputs = At_in.shape[1]
        
        assert num_inputs <= 1
        assert num_outputs <= 2

        Af_in = FFT(At_in[:,0], env.dt)
                
        splits = (env.f[0]-env.df, self.at[0]*env.f[-1])
        split1 = min(splits)
        split2 = max(splits)
        
        mask1 = np.ones(env.N)
        mask1[env.f <= split1] = 0; mask1[env.f > split2] = 0        
        mask2 = np.logical_not(mask1).astype('float')
        
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


