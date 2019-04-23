#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import square, sawtooth

from assets.functions import P, FFT, IFFT, RFSpectrum, PSD
plt.close('all')
#%%
def squarewave(t, amplitude, frequency, phase, duty):
    return amplitude * (square(2*np.pi*frequency*t + phase, duty)+1)/2

def sawwave(t, amplitude, frequency, phase, width):
    return amplitude * (sawtooth(2*np.pi*frequency*t + phase, width)+1)/2


#%%
N = 1000
t = np.linspace(0,100,N)
dt = t[1]-t[0]

f = np.fft.fftfreq(N, dt)
df = f[1] - f[0]

phasediff = 1

A1 = squarewave(t, 1,0.32, 0, 0.5)
A2 = sawwave(t, 1,0.12, phasediff, 0.5)

fig, ax = plt.subplots(2,1)

ax[0].plot(t, A1, label='A1', ls=":", color='blue')
ax[0].plot(t, A2, label='A2', ls="-", color='red')

RF1 = RFSpectrum(A1, dt)
RF2 = RFSpectrum(A2, dt)
ax[1].plot(RF1, label='A1', ls=":", color='blue')
ax[1].plot(RF2, label='A2', ls="-", color='red')

plt.show()

#%%
norm = np.sum(RF1) + np.sum(RF2)
metric = np.sum(np.abs(RF1 - RF2))/norm
print(metric)


