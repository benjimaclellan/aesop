import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
from scipy import signal
import time
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
from numpy import pi

plt.close('all')
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def fitness(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt):
    shifted = shift_function(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt)
    return np.sum(np.power(target-shifted,2))

def fitness_roll(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt):
    shifted = roll_function(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt)
    return np.sum(np.power(target-shifted,2))

def shift_function(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt):
    generated_rfft = np.fft.fft(generated)

    phase = np.angle(generated_rfft[target_harmonic_ind]/target_rfft[target_harmonic_ind])
    print('Detected phase is ',phase)
    
    shift = phase / ( target_harmonic * dt)
    shifted_rfft = generated_rfft * np.exp(-1j* shift * array)
    shifted = np.abs( np.fft.ifft(shifted_rfft) )
    return shifted


def roll_function(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt):
    
    generated_rfft = np.fft.rfft(generated)

    phase = np.angle(generated_rfft[target_harmonic_ind]/target_rfft[target_harmonic_ind])
    print('Detected phase is ',phase)
    
    shift = np.round(phase / (2 * pi * target_harmonic * dt)).astype('int')
    shifted = np.roll( generated, shift )
    return shifted

nu = 7e9
n_osc = 100
N = 2**13
cutoff = 100e9

t = np.linspace(0,n_osc / nu, N)

func = lambda t, phase: 0.5*(signal.sawtooth(2*np.pi*nu*t + phase)+1)
#func = lambda t, phase: 0.5*(signal.square(2*np.pi*nu*t + phase)+1)
#func = lambda t, phase: 0.5*(np.sin(2*np.pi*nu*t + phase)+1)
#func = lambda t, phase: np.sin(2*np.pi*nu*t + phase) + np.sin(4*np.pi*nu*t + 2*phase) + 0.1*np.sin(6*np.pi*nu*t + 4*phase)

func1 = lambda t, phase: 0.5*(signal.square(2*np.pi*nu*t + phase)+1)


target = func(t,0)
target_rfft = np.fft.fft(target)

dt = t[1]-t[0]
f = ( np.fft.fftfreq(N, dt) )
rf = np.fft.fftfreq(N,dt)
df = f[1]-f[0]

target_harmonic = nu
target_harmonic_ind = find_nearest(rf, target_harmonic)

phase_offset = np.random.uniform(0.4,np.pi-0.2)
generated = func(t,phase_offset)
#generated = func1(t,phase_offset)

print('Applied phase offset', phase_offset)


array = np.fft.fftshift( np.linspace(0,len(target_rfft)-1,len(target_rfft)) )/N
shifted = shift_function(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt)
#rolled = roll_function(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt)

plt.figure()
l1, = plt.plot(t,target, alpha=0.3, label='VT')
l2, = plt.plot(t,generated,ls=':', alpha=0.9,label='VX')
l3, = plt.plot(t,shifted,alpha=0.4,ls='--',label='shifted')
#plt.plot(t,rolled,alpha=0.4,ls='--',label='rolled')

plt.xlim(0,5/target_harmonic)
plt.legend()
plt.show()

fit = fitness(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt)
#fit_roll = fitness_roll(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt)

print('Fitness is ', fit)

scale = 10
for ii in range(0,scale):
    phase_offset = pi*ii/scale
    generated = func(t,phase_offset)
    
    shifted = shift_function(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt)
    fit = fitness(target, target_rfft, generated, array, target_harmonic, target_harmonic_ind, dt)

    l2.set_ydata(generated)
    l3.set_ydata(shifted)
    plt.show()
    plt.pause(0.5)
    
    print('Applied phase offset {}, fitness {}'.format(phase_offset, fit))

