#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A number of useful functions that are utilized through the simulation modules
"""

import pickle
import autograd.numpy as np


def power_(state):
    """Power of a signal, as the square of the absolute value
    """
    return np.power(np.abs(state), 2)


def psd_(state, dt, df, ax=0):
    """Power spectral density of a spectrum
    """
<<<<<<< HEAD
    tmp = np.power(np.abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(state, axes=ax), axis=ax), axes=ax) * dt), 2) / df
    print(f'psd_: {tmp}')
    return tmp
    # return np.power(np.abs(fft_(state, dt, ax)), 2) / df
=======
    return np.power(np.abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(state, axes=ax), axis=ax), axes=ax) * dt), 2) / df
>>>>>>> df2ce4016d3cb9da63bd7f4cdadd63ce1c810aaf


def fft_(state, dt, ax=0):
    """Proper Fast Fourier Transform for zero-centered vectors
    """
<<<<<<< HEAD
    tmp = np.fft.fft(state, axis=ax) * dt
    print(f'fft_: {tmp}')
    return tmp
    # return np.fft.fftshift(np.fft.fft(np.fft.fftshift(state, axes=ax), axis=ax), axes=ax) * dt
=======
    return np.fft.fft(state, axis=ax) * dt

>>>>>>> df2ce4016d3cb9da63bd7f4cdadd63ce1c810aaf

def fft_shift_(state, ax=0):
    """Proper Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.ifftshift(state, axes=ax)


def ifft_(state_f, dt, ax=0):
    """Proper Inverse Fast Fourier Transform for zero-centered vectors
    """
<<<<<<< HEAD
    tmp = np.fft.ifft(state_f, axis=ax) / dt
    print(f'ifft_: {tmp}')
    return tmp
    # return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(state_f, axes=ax), axis=ax), axes=ax) / dt
=======
    return np.fft.ifft(state_f, axis=ax) / dt
>>>>>>> df2ce4016d3cb9da63bd7f4cdadd63ce1c810aaf


def rfspectrum_(state, dt, ax=0):
    """Radio Frequency spectrum (ie spectrum off of a photodiode). Note that we use the real FFT
    """
    return np.abs(np.fft.rfft(np.power(np.abs(state), 2), axis=ax))


def rf_(power_trace, dt, ax=0):
    """Radio Frequency spectrum (ie spectrum off of a photodiode). Note that we use the real FFT
    """
    return np.abs(np.fft.rfft(power_trace, axis=ax))
