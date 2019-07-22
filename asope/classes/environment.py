import autograd.numpy as np
import peakutils
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from scipy.signal import square
import matplotlib.pyplot as plt

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


#%%
class OpticalField(object):
    """   
        Custom class for the optical field environment
    """
    def __init__(self, n_samples, **kwargs):
        self.n_samples = n_samples        
        self.__dict__.update(kwargs)
        
        self.custom_initialize()
        self.At = self.At0
        return
    
    
    def generate_time_frequency_arrays(self):
        # temporal domain
        t, dt = np.linspace(-self.window_t/2, self.window_t/2, self.n_samples, retstep=True) # time in s
        t = t.reshape((self.n_samples,1))
                  
        # frequency domain
        f, df = np.linspace(-self.n_samples/self.window_t/2, self.n_samples/self.window_t/2, self.n_samples, retstep=True)    # frequency in Hz
        f = f.reshape((self.n_samples, 1))

        
        self.t = t
        self.dt = dt
        self.f = f
#        self.f_rf = f[self.n_samples//2:]
        self.df = df
        return
        
                        
    def custom_initialize(self):
        """
            Define for each type of optical field
        """
        return
    
    
            
    def reset(self):
        """
            Resets the optical field to the initial
        """
        self.At = self.At0
        return
    
         
    def init_fitness(self):
        """
            Initialize fitness            
        """
        return 
    
    
    def fitness(self, At):
        """
            Wrapper function for the fitness function used. Here, the function to optimize can be changed without affecting the rest of the code
        """
        return 
    
    def compare(self, At):
        
        return
    
    def createnoise(self):
        """
            Noise model - not fully implemented
        """
        self.noise = np.random.normal(0, 0.05, np.shape(self.At0))
        return True
    
    def addnoise(self):
        """
            Injects the noise model on the pulse - not fully implemented
        """
        self.At0 += self.noise
        return True
    





#%%
class OpticalField_CW(OpticalField):
    """   
        Custom class for the continuous wave (CW) optical field environment
        
        Example:
            env = OpticalField_CW(n_samples=2**13+1, window_t=10e-9, peak_power=1)
            target_harmonic = 12e9
            env.init_fitness(lambda t: 0.5*(signal.square(2*np.pi*target_harmonic*t)+1), target_harmonic)
    """
        
    def custom_initialize(self):    
        self.generate_time_frequency_arrays()             
        self.At0 = self.peak_power * np.ones([self.n_samples, 1], dtype='complex')               
        self.add_noise = False
        return 
         
    def init_fitness(self, target, target_harmonic, normalize):
        
        self.target = target
        
        self.targetRF = RFSpectrum(self.target, self.dt)
        
        self.target_f = np.fft.fft(self.target, axis=0)
        self.scale_array = (np.fft.fftshift( np.linspace(0,len(self.target_f)-1,len(self.target_f)) )/ self.n_samples).reshape((self.n_samples, 1))
        self.target_harmonic = target_harmonic
        
        self.target_harmonic_ind = (self.target_harmonic/self.df).astype('int')
        
        self.normalize = normalize
        return 
    
    
    def fitness(self, At):
        """
            Wrapper function for the fitness function used. Here, the function to optimize can be changed without affecting the rest of the code
        """
        return self.waveform_temporal_overlap(At)
    
    def waveform_temporal_overlap(self, At):
  
        generated = P(At)        
        minval, maxval = np.min(generated), np.max(generated)
        if self.normalize:
            generated = (generated)/(maxval)
        
        shifted = self.shift_function(generated)
        
        overlap_integral = -np.sum(np.abs(self.target-shifted))
        total_amplitude = maxval-minval

        return overlap_integral, #total_amplitude        

    def shift_function(self, generated):
        rfft = np.fft.fft(generated, axis=0)
        phase = np.angle(rfft[self.target_harmonic_ind]/self.target_f[self.target_harmonic_ind])

        
        shift = phase / ( self.target_harmonic * self.dt )
        rfft = rfft * np.exp(-1j* shift * self.scale_array)
        shifted = np.abs( np.fft.ifft(rfft, axis=0) )
        return shifted

#    def compare(self, At):
#        generated = self.shift_function( P(At) )
#        fig, ax = plt.subplots(1, 1, figsize=(8, 10), dpi=80)
#        ax.plot(self.t, self.target, label='Target')
#        ax.plot(self.t, generated, label='Generated')    
#        plt.legend()
#        
#        return
#%%
class OpticalField_Pulse(OpticalField):
    """   
        Custom class for the pulse train optical field environment
        Example:
            env = OpticalField_Pulse(n_samples=2**12, profile='gaussian', pulse_width=50e-12, f_rep=100e6, n_pulses=30, peak_power=1)
            env.init_fitness(p=2, q=1)
    """
       
    def custom_initialize(self): 
        self.window_t = 1/self.f_rep * self.n_pulses        
        self.generate_time_frequency_arrays()
        
        # create initial train of Gaussian pulses
        if self.profile == 'gaussian':
            self.At0 = np.zeros([self.n_samples, 1], dtype='complex')
            for i_pulse in range(0,self.n_pulses+1):
                self.At0 += np.exp(-0.5 * (np.power((self.t+(self.t[0] + self.window_t*(i_pulse/(self.n_pulses))))/self.pulse_width, 2)))
        
        # create initial train of sech2 pulses
        elif self.profile == 'sech':
            self.At0 = np.zeros([self.n_samples, 1], dtype='complex')
            for i_pulse in range(0,self.n_pulses+1):
                self.At0 += sech((self.t+(self.t[0] + self.window_t*(i_pulse/(self.n_pulses))))/self.pulse_width)

        else:
            raise ValueError
            
        # scale by peak_power power
        self.At0 *= np.sqrt(self.peak_power)
        return     

    
         
    def init_fitness(self, p, q):
        self.p = p
        self.q = q
        
        return 
    
    
    def fitness(self, At):
        """
            Wrapper function for the fitness function used. Here, the function to optimize can be changed without affecting the rest of the code
        """
        return self.TalbotEffect(At)
    
  
  
    def TalbotEffect(self, At):
        """
            One possible fitness function, looking to increase or decrease the repetition rate of the pulse train. We use the harmonic frequencies in the RF spectrum to measure this
        """
        
        # one value to optimize is the peak power
        PAt = P(At)
        X1 = np.max(PAt) - self.p/self.q
        fitness2 = supergaussian(X1,1,2)

        # find the harmonic frequencies in the RF domain
        peakinds = peakutils.indexes(RFSpectrum(At, self.dt))
        
        # measure the distance between RF spectrum peaks
        peakf_dist = np.mean( self.df * np.diff(peakinds) )
        
        # our target is increased/decreased rep-rate from the original pulse
        peakf_distTarget = self.f_rep / (self.p/self.q)
        
        # sometimes no peaks are found (bad individual with no real structure), so set fitness to 0
        if len(peakinds) <= 1:
            fitness1 = 0 
        
        else: 
            X = np.power( peakf_distTarget - peakf_dist, 2)
            fitness1 = supergaussian(X,1,1)

        return (fitness1, fitness2)
    


