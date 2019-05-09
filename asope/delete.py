from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def random_bit_pattern(n_samples, n_bits, pattern_frequency, specific_ratio):
    if specific_ratio == None:
        bit_sequence = np.random.random_integers(0,1,n_bits)
    else:
        bit_sequence = np.array([0] * round(n_bits * specific_ratio) + [1] * round(n_bits * (1-specific_ratio)))
        np.random.shuffle(bit_sequence)
    
    pattern_samples = (1/pattern_frequency/dt).astype('int')
    tmp = np.repeat(bit_sequence, pattern_samples)
    pattern = np.tile(tmp, np.ceil(n_samples/pattern_samples).astype('int'))[0:n_samples]
    return pattern, bit_sequence



n_samples = 2**12
t = np.linspace(0, 100, n_samples)
dt = t[1]-t[0]
#y = signal.sweep_poly(t, [1,1])

f0 = 1 # Hz
t1 = t[len(t)//2]
f1 = f0/2

y = signal.chirp(t, f0, t1, f1, 'quadratic')
y = np.cos(f0*t)

n_bits = 10
specific_ratio = 5/n_bits
pattern_frequency = 1

pattern, bit_sequence = random_bit_pattern(n_samples, n_bits, pattern_frequency, specific_ratio)

print(bit_sequence)

plt.close('all')
plt.plot(t,pattern)
plt.show()