import numpy as np
import peakutils
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from scipy.signal import square
from assets.config import FITNESS_VARS

def sech(x):
    return 1/np.cosh(x, dtype='complex')

def dsigmoid(X, A):
    return 4*(1/(1+np.exp(-X*X/A)))*(1-1/(1+np.exp(-X*X/A)))

def clippedfitness(X):
    return max([0,X])

def gaussian(X,A):
    return np.exp(-X*X/A)

def supergaussian(X, A, m):
    return np.exp(-np.power(X*X/A,m))



def fitness_shift(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt):
    shifted = shift_function(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt)
    return -np.sum(np.abs(target-shifted)), shifted

def shift_function(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt):
    generated_rfft = np.fft.fft(generated, axis=0)
    phase = np.angle(generated_rfft[target_harmonic_ind]/target_rfft[target_harmonic_ind])
#    print('Detected phase is ',phase)
#    generated_rfft=np.expand_dims(generated_rfft, 2)
#    assert generated_rfft.shape == array.shape
    
#    print(generated_rfft.shape, phase.shape, target_rfft.shape, array.shape)
    shift = phase / ( target_harmonic * dt )
    shifted_rfft = generated_rfft * np.exp(-1j* shift * array)
    shifted = np.abs( np.fft.ifft(shifted_rfft, axis=0) )
    return shifted


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


class PulseEnvironment(object):
    """   
        Custom class for the pulse environment, storing all the details of an input pulse train
    """
    def __init__(self):
        
        self.profile = FITNESS_VARS['profile']
                
#        # pulse parameters
#        N = 2**13+1        # number of points in the simulation [] 
#        T = 500e-9       # pulse width [s]
#        res = 2**17      # resolution the agent sees from 'oscilliscope' [2^bits]
#        n_pulses = 10   # number of pulses in simulation window []
#        f_rep = 1e9    # repetition rate of the pulse train [Hz]
#        peakP = 1        # peak power [W]
#        window = 1/f_rep * n_pulses
        
        
        # pulse parameters
        N = 2**13+1        # number of points in the simulation [] 
        T = 500e-9       # pulse width [s]
        res = 2**17      # resolution the agent sees from 'oscilliscope' [2^bits]
        n_pulses = 100   # number of pulses in simulation window []
        f_rep = 1e6    # repetition rate of the pulse train [Hz]
        peakP = 1        # peak power [W]
        
        window = 50 * 1/FITNESS_VARS['target_harmonic']
        
        # temporal domain
        
        t = np.linspace(-window/2, window/2, N).reshape(N, 1)     # time in s
        dt = (t[1] - t[0])[0]
                  
        # frequency domain
        f = np.linspace(-N/2, N/2-1, N).reshape(N, 1) / window    # frequency in Hz
        df = (f[1] - f[0])[0]
        
        # create initial train of Gaussian pulses
        if self.profile == 'gauss':
            At0 = np.zeros([N, 1], dtype='complex')
            for i_pulse in range(0,n_pulses+1):
                At0 += np.exp(-0.5 * (np.power((t+(t[0] + window*(i_pulse/(n_pulses))))/T, 2)))
        
        # create initial train of sech2 pulses
        elif self.profile == 'sech':
            At0 = np.zeros([N, 1], dtype='complex')
            for i_pulse in range(0,n_pulses+1):
                At0 += sech((t+(t[0] + window*(i_pulse/(n_pulses))))/T)

                # create initial cw wave (carrier removed)
        elif self.profile == 'cw':
            At0 = np.ones([N, 1], dtype='complex')
        
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
        self.f_rf = f[N//2:]
        self.df = df
        
        self.At0 = At0
        self.Af0 = Af0
        
        self.reset()
        
        # intialize variables for the fitness function
        self.init_fitness()
        return
    
            
    def reset(self):
        """
            Resets the current pulse to the intial
        """
        self.At = self.At0
        self.Af = self.Af0
        return
    
         
    def init_fitness(self):
        self.target = FITNESS_VARS['func'](self.t)
        self.targetRF = RFSpectrum(self.target, self.dt)
        
        self.target_rfft = np.fft.fft(self.target, axis=0)
        self.array = (np.fft.fftshift( np.linspace(0,len(self.target_rfft)-1,len(self.target_rfft)) )/ self.N).reshape((self.N, 1))
        self.target_harmonic = FITNESS_VARS['target_harmonic']
        
#        rf = np.fft.fftfreq(self.N,self.dt)

#        self.target_harmonic_ind = find_nearest(rf, self.target_harmonic)
        self.target_harmonic_ind = (self.target_harmonic/self.df).astype('int')
        
        
        return 
    
    
    def fitness(self, At):
        """
            Wrapper function for the fitness function used. Here, the function to optimize can be changed without affecting the rest of the code
        """
        return self.PhotonicAWG(At)
#        return self.TalbotEffect(At)
    
    def PhotonicAWG(self, At):
        
#        
#        RF = RFSpectrum(At, self.dt)
##        norm = np.sum(RF) + np.sum(self.targetRF)
#        if RF.shape != self.targetRF.shape:
#            RF=np.expand_dims(RF, 2)
#        metric = -np.sum(np.abs(RF - self.targetRF),axis=0)[0]
#        assert RF.shape == self.targetRF.shape
        
        
#        RF = RFSpectrum(At, self.dt)
#        norm = np.sum(RF) + np.sum(self.targetRF)
#        if RF.shape != self.targetRF.shape:
#            RF=np.expand_dims(RF, 2)
#        metric = -np.sum(np.abs(RF - self.targetRF),axis=0)[0]
#        metric = -np.sum(np.abs(np.fft.rfft(P(At)) - np.fft.rfft(self.target)))
            
            
        PAt = P(At)
#        minPAt, maxPAt = np.min(PAt), np.max(PAt)
#        generated = (PAt-minPAt)/(maxPAt-minPAt)
        generated = (PAt)

        
#        metric = -np.sum(np.abs((PAt-minPAt)/(maxPAt-minPAt) - self.target))
#        assert RF.shape == self.targetRF.shape
        
        fit, shifted = fitness_shift(self.target, self.target_rfft, generated, self.array, self.target_harmonic, self.target_harmonic_ind, self.dt)
        
#        self.shifted = shifted
        
        return (fit),# (maxPAt-minPAt))
    
    
    
    def TalbotEffect(self, At):
        """
            One possible fitness function, looking to increase or decrease the repetition rate of the pulse train. We use the harmonic frequencies in the RF spectrum to measure this
        """
        p = FITNESS_VARS['p']
        q = FITNESS_VARS['q']
        
        # one value to optimize is the peak power
        PAt = P(At)
        X1 = np.max(PAt) - p/q
        fitness2 = supergaussian(X1,1,2)

        # find the harmonic frequencies in the RF domain
        peakinds = peakutils.indexes(RFSpectrum(At, self.dt))
        
        # measure the distance between RF spectrum peaks
        peakf_dist = np.mean( self.df * np.diff(peakinds) )
        
        # our target is increased/decreased rep-rate from the original pulse
        peakf_distTarget = self.f_rep / (p/q)
        
        # sometimes no peaks are found (bad individual with no real structure), so set fitness to 0
        if len(peakinds) <= 1:
            fitness1 = 0 
        
        else: 
            X = np.power( peakf_distTarget - peakf_dist, 2)
            fitness1 = supergaussian(X,1,1)

#        fitness1 = P(At[self.N//2])


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
    
    