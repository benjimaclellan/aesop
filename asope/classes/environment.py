import autograd.numpy as np
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum, LowPassRF
import scipy as sp
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
        self.field = self.field0
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
        self.field = self.field0
        return
    
         
    def init_fitness(self):
        """
            Initialize fitness            
        """
        return 
    
    
    def fitness(self, field):
        """
            Wrapper function for the fitness function used. Here, the function to optimize can be changed without affecting the rest of the code
        """
        return 
    
    def compare(self, field):
        
        return

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
        self.field0 = self.peak_power * np.ones([self.n_samples, 1], dtype='complex')
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
    
    
    def fitness(self, field, exp=None):
        """
            Wrapper function for the fitness function used. Here, the function to optimize can be changed without affecting the rest of the code
        """
        # return self.average_lowfreq_rf(field),
        return self.waveform_temporal_overlap(field),


    def average_lowfreq_rf(self, field):
        # filtered = LowPassRF(field, self, 1e9, self.dt)
        # return (np.mean(np.abs(filtered)))
        return (np.mean(np.abs(field)))

    def waveform_temporal_overlap(self, field):
  
        generated = P(field)
        minval, maxval = np.min(generated), np.max(generated)
        if self.normalize:
            generated = (generated)/(maxval)
        
        shifted = self.shift_function(generated)
        
        overlap_integral = -np.sum(np.abs(self.target-shifted))
        total_amplitude = maxval-minval

        return overlap_integral

    def shift_function(self, generated):
        rfft = np.fft.fft(generated, axis=0)
        phase = np.angle(rfft[self.target_harmonic_ind]/self.target_f[self.target_harmonic_ind])

        
        shift = phase / ( self.target_harmonic * self.dt )
        rfft = rfft * np.exp(-1j* shift * self.scale_array)
        shifted = np.abs( np.fft.ifft(rfft, axis=0) )
        return shifted


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
            self.field0 = np.zeros([self.n_samples, 1], dtype='complex')
            for i_pulse in range(0,self.n_pulses+1):
                self.field0 += np.exp(-0.5 * (np.power((self.t+(self.t[0] + self.window_t*(i_pulse/(self.n_pulses))))/self.pulse_width, 2)))
        
        # create initial train of sech2 pulses
        elif self.profile == 'sech':
            self.field0 = np.zeros([self.n_samples, 1], dtype='complex')
            for i_pulse in range(0,self.n_pulses+1):
                self.field0 += sech((self.t+(self.t[0] + self.window_t*(i_pulse/(self.n_pulses))))/self.pulse_width)

        else:
            raise ValueError
            
        # scale by peak_power power
        self.field0 *= np.sqrt(self.peak_power)
        return     

    
         
    def init_fitness(self, p, q):
        self.p = p
        self.q = q
        
        return 
    
    
    def fitness(self, field):
        """
            Wrapper function for the fitness function used. Here, the function to optimize can be changed without affecting the rest of the code
        """
        return np.mean(P(field)),

    def Sensor(self, field):
        return np.mean(P(field)),
  
    def TalbotEffect(self, field):
        """
            One possible fitness function, looking to increase or decrease the repetition rate of the pulse train. We use the harmonic frequencies in the RF spectrum to measure this
        """
        
        # one value to optimize is the peak power
        PAt = P(field)
        freq_target = int(round(self.f_rep / self.df)/(self.p / self.q))
        X2 = np.abs(np.max(PAt) - self.p / self.q)
#        X1 = np.sum(RFSpectrum(field, self.dt)[freq_target-1:freq_target+1])
        X1 = (RFSpectrum(field, self.dt)[freq_target])  #/ X2

        return X1


# %%
class OpticalField_PPLN(OpticalField):
    """
        Custom class for the continuous wave (CW) optical field environment

        Example:
            env = OpticalField_CW(n_samples=2**13+1, window_t=10e-9, peak_power=1)
            target_harmonic = 12e9
            env.init_fitness(lambda t: 0.5*(signal.square(2*np.pi*target_harmonic*t)+1), target_harmonic)
    """

    def custom_initialize(self):
        self.generate_time_frequency_arrays()
        self.field0 = self.peak_power * np.ones([self.n_samples, 1], dtype='complex')
        self.add_noise = False
        return

    def init_fitness(self):
        return

    def fitness(self, field, exp=None):
        return self.fsr(field),

    def fsr(self, field):
        return 0

