import numpy as np
import peakutils


class PulseEnvironment(object):
    """   
    A class that stores all our simulation parameters, time and frequency domain of the optical signal, etc. This will (could) reduce the number of arguments passed into different function calls
    """
    def __init__(self, p, q):
        self.p = p
        self.q = q
        
        
        N = 2**14        # number of points in the simulation [] 
        T = 50e-12      # pulse width [s]
        res = 2**17      # resolution the agent sees from 'oscilliscope' [2^bits]
        n_pulses =  10#30   # number of pulses in simulation window []
        f_rep = 100e6   # repetition rate of the pulse train [Hz]
        peakP = 1           # peak power [W]
        
        window = 1/f_rep * n_pulses
        (t, dt) = np.linspace(-window/2, window/2, N, retstep=True)     # time in s
                
        f = np.linspace(-N/2, N/2-1, N) / window    # frequency in Hz
        df = f[1] - f[0]
        
        At0 = np.zeros(N).astype('complex')
        for i_pulse in range(0,n_pulses+1):
            At0 += np.exp(-0.5 * (np.power((t+(t[0] + window*(i_pulse/(n_pulses))))/T, 2)))


        At0 *= np.sqrt(peakP)
        Af0 = self.FFT(At0, dt)
        
        self.add_noise = False
        
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
        self.At = self.At0
        self.Af = self.Af0
    
    def P(self, At):
        return np.power( np.abs( At ), 2)
    
    def PSD(self, Af, df):
        return np.power( np.abs( Af ), 2) / df

    def FFT(self, At, dt):   # proper Fast Fourier Transform
        return np.fft.fftshift(np.fft.fft(np.fft.fftshift(At)))*dt

    def IFFT(self, Af, dt):  # proper Inverse Fast Fourier Transform
        return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Af)))/dt
    
    
    def RFSpectrum(self, At, dt):
#        return np.abs( self.FFT(np.power( np.abs( At ), 2), dt) )
        return np.abs(np.fft.rfft(np.power( np.abs( At ), 2)))
        
    def CreateNoise(self):
#        noise = (np.sqrt(2*np.pi*spread*spread)) * np.random.normal(mean, spread, self.f.shape) #+ 1j * np.random.uniform(-np.pi, np.pi, self.f.shape)
#        noise = 0.001*(1/self.df) * np.random.uniform(-1, 1, self.f.shape) * np.exp( 1j * np.random.uniform(-np.pi, np.pi, self.f.shape))
#        noise = np.exp( 1j * np.pi * 4 * np.random.normal(-1, 1, self.f.shape))
        noise = np.random.uniform(0.9, 1.1, self.f.shape) * np.exp( 1j * np.random.normal(0, 0.7, self.f.shape))
        return noise
    
    def AddNoise(self, noise):
        self.Af = self.Af * noise
        self.At = self.IFFT(self.Af, self.dt)
        return
            
    def fitness(self, At):
        fitness = self.TalbotEffect(At, self.p, self.q)
        return fitness
    
    def TalbotEffect(self, At, p, q):
        fitness2 = np.max(self.P(At))
            
        peakinds = peakutils.indexes(self.RFSpectrum(At, self.dt))
        
        peakf_dist = np.mean( self.df * np.diff(peakinds) )
        peakf_distTarget = self.f_rep / (p/q)
        
        if len(peakinds) <= 1:
            fitness1 = 0 #-99999999
        else: 
            fitness1 = 1 - np.power( peakf_distTarget - peakf_dist, 2)
        return (fitness1, fitness2)
        
    
    