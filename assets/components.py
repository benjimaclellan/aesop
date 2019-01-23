import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from copy import copy
#from assets.functions import splitindices

#plt.close("all")

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
        
    def crossover(self):
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
        self.UPPER = [20000]
        self.LOWER = [0]
        self.DTYPE = ['float']
        self.DSCRTVAL = [None]
        self.FINETUNE_SKIP = 0
    
    def simulate(self, env, visualize=False):
        fiber_len = self.at[0]   
        D = np.zeros(env.f.shape).astype('complex')
        for n in range(0, len(self.beta)):    
            D += self.beta[n] * np.power(2*np.pi*env.f, n+2) / np.math.factorial(n+2)
        D = -1j * D
        env.Af = np.exp(fiber_len * D) * env.Af
        env.At = env.IFFT( env.Af, env.dt )
        
        if visualize:
            self.lines = ((None),)
            
        return
    
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
        self.FINETUNE_SKIP = 0
        
    
    def simulate(self, env, visualize=False):
        M = self.at[0]       # amplitude []
        NU = self.at[1]      # frequency [Hz]
        PHI = self.at[2]     # phase offset [rad]
        phase = (M/2)*np.cos(2*np.pi* NU * env.t + PHI) + M/2
        
        ## Apply phase modulator in time-domain, and update frequency
        env.At = env.At * np.exp(1j * phase)                
        env.Af = env.FFT(env.At, env.dt)
        
        if visualize:
            self.lines = (('t',phase),)
        
        return

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
        self.FINETUNE_SKIP = 1
 
    
    def simulate(self, env, visualize=False):
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
        
        env.At = env.At * np.exp(1j * phase) * (.95**nlevels) #loss here
        env.Af = env.FFT(env.At, env.dt)
        
        if visualize:
            self.lines = (('t',phase),)
        
        return
    
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
        self.FINETUNE_SKIP = 0
 
    
    def simulate(self, env, visualize = False):
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
        
            
        env.Af = ampmask * np.exp(1j * phasemask) * env.Af
        env.At = env.IFFT( env.Af, env.dt )

        if visualize:
            self.lines = (('f',ampmask), ('f',phasemask))
        
        return
    
    def windowmask(self, mask, N, n):
        totalbins = int(np.ceil( N / n ) )
        center = N // 2
        bins2center = int( np.floor( center / n ) )
        offset = abs((bins2center*n) - center) - n//2
#        print(N,n,offset)
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





class BeamSplitter(Component):
    _num_instances = count(0)
    def datasheet(self):
        self.ratio = 0.5
        self.N_PARAMETERS = 1
        self.UPPER = [0.5]
        self.LOWER = [0.5]
        self.DTYPE = ['float']
        self.DSCRTVAL = [None]
        self.FINETUNE_SKIP = 0
    
    def simulate(self, env, visualize=False):
        raise ValueError()        
        if visualize:
            self.lines = ((None),)    
        return
    
    def newattribute(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at
    
    def mutate(self):
        at = [self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])]
        self.at = at
        return at
