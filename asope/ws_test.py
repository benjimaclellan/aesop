import numpy as np
import matplotlib.pyplot as plt
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from numpy import pi

plt.close('all')

vals = [0,1,0,1,0] + [0,pi/2,pi,3*pi/2,2*pi]
env = PulseEnvironment()

ampvals = vals[0:5]
phasevals = vals[5:]
phase = np.zeros_like(env.f)
amp = np.ones_like(env.f)

res = 12e9

n = np.floor(env.N/((1/env.dt)/res)).astype('int')
tmp = np.ones((n,1))

a = [i*tmp for i in ampvals]
p = [i*tmp for i in phasevals]


amp1 = np.concatenate(a)
phase1 = np.concatenate(p)

left = np.floor((env.N - amp1.shape[0])/2).astype('int')
right = env.N - np.ceil((env.N - amp1.shape[0])/2).astype('int')

phase[left:right] = phase1
amp[left:right] = amp1

Af = amp * np.exp(1j * phase) * FFT(env.At, env.dt)
At = IFFT( Af, env.dt )



plt.plot(env.f, phase, ls=':',  alpha=0.5, label='phase')
plt.plot(env.f, amp, ls='--', alpha=0.5, label='amp')
plt.plot(env.f, PSD(Af, env.df)/np.max(PSD(Af, env.df)),label='Af')
plt.legend()