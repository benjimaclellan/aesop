#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A number of useful functions that are utilized through the simulation modules
"""

import pickle
import autograd.numpy as np


def P(field):
    """Power of a signal, as the square of the absolute value
    """
    return np.power(np.abs(field), 2)


def PSD(field, dt, df, ax=0):
    """Power spectral density of a spectrum
    """
    return np.power(np.abs(FFT(field, dt, ax)), 2) / df


def FFT(field, dt, ax=0):
    """Proper Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(field, axes=ax), axis=ax), axes=ax) * dt


def IFFT(Af, dt, ax=0):
    """Proper Inverse Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Af, axes=ax), axis=ax), axes=ax) / dt


def RFSpectrum(field, dt, ax=0):
    """Radio Frequency spectrum (ie spectrum off of a photodiode). Note that we use the real FFT
    """
    return np.abs(np.fft.rfft(np.power(np.abs(field), 2), axis=ax))

