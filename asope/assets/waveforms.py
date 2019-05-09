import numpy as np

def random_bit_pattern(n_samples, n_bits, pattern_frequency, specific_ratio, dt):
    if specific_ratio == None:
        bit_sequence = np.random.random_integers(0,1,n_bits)
    else:
        bit_sequence = np.array([0] * round(n_bits * specific_ratio) + [1] * round(n_bits * (1-specific_ratio)))
        np.random.shuffle(bit_sequence)
    
    pattern_samples = (1/n_bits/pattern_frequency/dt).astype('int')
    print(pattern_samples, n_samples)
    tmp = np.repeat(bit_sequence, pattern_samples)
    pattern = np.tile(tmp, np.ceil(n_samples/pattern_samples).astype('int'))[0:n_samples]
    pattern = pattern.reshape((n_samples, 1))
    assert pattern.ndim == 2
    
    return pattern, bit_sequence
