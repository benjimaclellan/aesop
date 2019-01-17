import numpy as np
#PYTHONPATH = '/home/benjamin/Documents/INRS - Code/ASOPE_V2_SingleSetup'
import sys
sys.path.append('/home/benjamin/Documents/INRS - Code/ASOPE_V2_SingleSetup')
from environments.environment_pulse import PulseEnvironment
import matplotlib.pyplot as plt


class Component(object):
    def __init__(self, id, type):
        self.name = str(type) + str(id)
        self.id = id
        self.type = type
        self.datasheet()
        
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
        
    def randomattribute(low=0.0, high=1.0, dtypes='float', dscrtval=None):
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
    def datasheet(self):
        self.beta = [2e-20]
        self.Tr = 0.5
        self.gamma = 100
        self.N_PARAMETERS = 1
        self.UPPER = [20000]
        self.LOWER = [0]
        self.DTYPE = ['float']
        self.DSCRTVAL = [None]
        self.FINETUNE_SKIP = 0
    
    def simulate(self, env, values):
        fiber_len = values[0]   
        D = np.zeros(env.f.shape).astype('complex')
        for n in range(0, len(self.beta)):    
            D += self.beta[n] * np.power(2*np.pi*env.f, n+2) / np.math.factorial(n+2)
        D = -1j * D
        env.Af = np.exp(fiber_len * D) * env.Af
        env.At = env.IFFT( env.Af, env.dt )
        return
    
    def newattribute(self):
        at = self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])
        return at
    
    def mutate(self, ind):
        ind[0] = self.randomattribute(self.LOWER[0], self.UPPER[0], self.DTYPE[0], self.DSCRTVAL[0])
        
    def crossover(self, ind1, ind2):
        
# ----------------------------------------------------------
"""

"""
class PhaseModulator(Component):
    def datasheet(self):
        self.vpi = 1
        self.N_PARAMETERS = 3
        self.UPPER = [10, 200e6, 2*np.pi]   # max shift, frequency, phase offset
        self.LOWER = [0, 1e6, 0]
        self.DTYPE = ['float', 'float', 'float']
        self.DSCRTVAL = [None, 1e6, None]
        self.FINETUNE_SKIP = 0
        
    
    def simulate(self, env, values):
        M = values[0]       # amplitude []
        NU = values[1]      # frequency [Hz]
        PHI = values[2]     # phase offset [rad]
        dPhase = (M/2)*np.cos(2*np.pi* NU * env.t + PHI) + M/2
        
        ## Apply phase modulator in time-domain, and update frequency
        env.At = env.At * np.exp(1j * dPhase)                
        env.Af = env.FFT(env.At, env.dt)
        return


# ----------------------------------------------------------
"""

"""
class AWG(Component):
    def datasheet(self):
        self.N_PARAMETERS = 2
        self.UPPER = [8] + [2*np.pi]   # max shift, frequency, phase offset
        self.LOWER = [1] + [0]
        self.DTYPE = ['int', 'float']
        self.DSCRTVAL = [1, None]
        self.FINETUNE_SKIP = 1
 
    
    def simulate(self, env, values):
        nlevels = values[0]
        phasevalues = values[1:]
        
        timeblock = np.round(1/env.dt/env.f_rep).astype('int')
        tmp = np.ones(timeblock)
        oneperiod = np.array([]).astype('float')
        
        for i in range(0,nlevels):
            oneperiod = np.concatenate((oneperiod, tmp*phasevalues[i]))            
        phasetmp = np.tile(oneperiod, np.ceil(len(env.t)/len(oneperiod)).astype('int') )

        shift1 = timeblock//2
        
        phasetmp = phasetmp[shift1:]
        phase = phasetmp[0:len(env.t)] 
        
        env.At = env.At * np.exp(1j * phase)
        env.Af = env.FFT(env.At, env.dt)
        return
    

env = PulseEnvironment()

c = PhaseModulator(1,'phasemodulator')
c.simulate(env, [10, 1e8, 0])
print(c.name)

fig, ax = plt.subplots(2, 1, figsize=(8, 10), dpi=80)
    
ax[0].set_xlabel('Time (ps)')
ax[0].set_ylabel('Power [arb]')
ax[1].set_xlabel('Frequency (THz)')
ax[1].set_ylabel('PSD [arb]')

ax[0].plot(env.t/1e-12, env.P(env.At0), label='initial')
ax[1].plot(env.f/1e12, env.PSD(env.Af0, env.df))

ax[0].plot(env.t/1e-12, env.P(env.At), label='final')
ax[1].plot(env.f/1e12, env.PSD(env.Af, env.df))