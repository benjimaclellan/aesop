import numpy as np
import peakutils
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum


def sech(x):
#    np.seterr('ignore')
    return 1/np.cosh(x, dtype='complex')

class PulseEnvironment(object):
    """   
        Custom class for the pulse environment, storing all the details of an input pulse train
    """
    def __init__(self, p, q, profile='gauss'):
        
        # the Talbot ratio, used in the fitness function
        self.p = p
        self.q = q
        
        # pulse parameters
        N = 2**14        # number of points in the simulation [] 
        T = 50e-12       # pulse width [s]
        res = 2**17      # resolution the agent sees from 'oscilliscope' [2^bits]
        n_pulses =  10   # number of pulses in simulation window []
        f_rep = 100e6    # repetition rate of the pulse train [Hz]
        peakP = 1        # peak power [W]
        
        # temporal domain
        window = 1/f_rep * n_pulses
        (t, dt) = np.linspace(-window/2, window/2, N, retstep=True)     # time in s
                
        # frequency domain
        f = np.linspace(-N/2, N/2-1, N) / window    # frequency in Hz
        df = f[1] - f[0]
        
        # create initial train of Gaussian pulses
        if profile == 'gauss':
            At0 = np.zeros(N).astype('complex')
            for i_pulse in range(0,n_pulses+1):
                At0 += np.exp(-0.5 * (np.power((t+(t[0] + window*(i_pulse/(n_pulses))))/T, 2)))
        
        # create initial train of sech2 pulses
        elif profile == 'sech':
            At0 = np.zeros(N).astype('complex')
            for i_pulse in range(0,n_pulses+1):
                At0 += sech((t+(t[0] + window*(i_pulse/(n_pulses))))/T)

                # create initial cw wave (carrier removed)
        elif profile == 'cw':
            At0 = np.ones(N).astype('complex')
        
        else:
            raise ValueError
            
        # scale by peak power
        At0 *= np.sqrt(peakP)
        Af0 = FFT(At0, dt)
        
        # noise is not fully implemented yet
        self.add_noise = False
        
        # save details to class
        self.N = N
        self.T = T
        self.peakP = peakP
        self.res = res
        self.n_pulses = n_pulses
        self.f_rep = f_rep
        self.window = window
        
        self.t = t
        self.dt = dt
        self.f = f
        self.df = df
        
        self.At0 = At0
        self.Af0 = Af0
        
        self.reset()
        return
    
            
    def reset(self):
        """
            Resets the current pulse to the intial
        """
        self.At = self.At0
        self.Af = self.Af0
        return
    
            
    def fitness(self, At):
        """
            Wrapper function for the fitness function used. Here, the function to optimize can be changed without affecting the rest of the code
        """
        
        # currently, we use the Talbot effect as the fitness - either increasing or decreasing the repetition rate of the pulse train by p/q
        fitness = self.TalbotEffect(At, self.p, self.q)
        return fitness
    
    def TalbotEffect(self, At, p, q):
        """
            One possible fitness function, looking to increase or decrease the repetition rate of the pulse train. We use the harmonic frequencies in the RF spectrum to measure this
        """
        
        # one value to optimize is the peak power
#        fitness2 = np.max(P(At))
        PAt = P(At)
#        fitness2 = np.max(PAt) #* np.std((PAt))
        dPAt = np.diff(PAt)
        fitness2 = np.max(PAt) * ((dPAt[:-1] * dPAt[1:]) < 0).sum() / self.N       
        
        # find the harmonic frequencies in the RF domain
        peakinds = peakutils.indexes(RFSpectrum(At, self.dt))
        
        # measure the distance between RF spectrum peaks
        peakf_dist = np.mean( self.df * np.diff(peakinds) )
        
        # our target is increased/decreased rep-rate from the original pulse
        peakf_distTarget = self.f_rep / (p/q)
        
        # sometimes no peaks are found (bad individual with no real structure), so set fitness to 0
        if len(peakinds) <= 1:
            fitness1 = 0 #-99999999
        # if there are peaks in the RF spectrum, define the fitness function here, based on how far we are from the target
        else: 
            fitness1 = 1 - np.power( peakf_distTarget - peakf_dist, 2)

        return (fitness1, fitness2)
    
    # --------------------------------------
    def CreateNoise(self):
        """
            Noise model - not fully implemented
        """
        noise = np.random.uniform(0.9, 1.1, self.f.shape) * np.exp( 1j * np.random.normal(0, 0.7, self.f.shape))
        return noise
    
    def AddNoise(self, noise):
        """
            Injects the noise model on the pulse - not fully implemented
        """
        self.Af = self.Af * noise
        self.At = IFFT(self.Af, self.dt)
        return
    
    