#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A number of useful functions that are utilized through the simulation modules
"""

import autograd.numpy as np
from numpy import unwrap

def power_(state):
    """Power of a signal, as the square of the absolute value
    """
    return np.power(np.abs(state), 2)


def psd_(state, dt, df, ax=0):
    """Power spectral density of a spectrum
    """
    return np.power(np.abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(state, axes=ax), axis=ax), axes=ax) * dt), 2) / df


def fft_(state, dt, ax=0):
    """Proper Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fft(state, axis=ax) * dt


def ifft_shift_(state, ax=0):
    """Proper Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.ifftshift(state, axes=ax)


def fft_shift_(state, ax=0):
    """Proper Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fftshift(state, axes=ax)

def ifft_(state_f, dt, ax=0):
    """Proper Inverse Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.ifft(state_f, axis=ax) / dt

def phase_(state, ax=0):
    return unwrap(np.angle(state), axis=ax)

def phase_spectrum_(state, dt, ax=0):
    return unwrap(np.angle(fft_shift_(fft_(state, dt))), axis=ax)

def rfspectrum_(state, dt, ax=0):
    """Radio Frequency spectrum (ie spectrum off of a photodiode). Note that we use the real FFT
    """
    return np.abs(np.fft.rfft(np.power(np.abs(state), 2), axis=ax))


def rf_(power_trace, dt, ax=0):
    """Radio Frequency spectrum (ie spectrum off of a photodiode). Note that we use the real FFT
    """
    return np.abs(np.fft.rfft(power_trace, axis=ax))

def dB_to_amplitude_ratio(loss_db):
    return 10.0 ** (loss_db/20.0)
