#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A number of useful functions that are utilized through the simulation modules
"""

import pickle
import autograd.numpy as np
import matplotlib.pyplot as plt
import numpy

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


def ifft_(state_f, dt, ax=0):
    """Proper Inverse Fast Fourier Transform for zero-centered vectors
    """
    return np.fft.ifft(state_f, axis=ax) / dt


def rfspectrum_(state, dt, ax=0):
    """Radio Frequency spectrum (ie spectrum off of a photodiode). Note that we use the real FFT
    """
    return np.abs(np.fft.rfft(np.power(np.abs(state), 2), axis=ax))


def rf_(power_trace, dt, ax=0):
    """Radio Frequency spectrum (ie spectrum off of a photodiode). Note that we use the real FFT
    """
    return np.abs(np.fft.rfft(power_trace, axis=ax))


def get_pulse_train(t, pulse_rep_t, pulse):
    """

    :param t:
    :param pulse_rep_t:
    :param pulse:
    :return:
    """
    # def sech(t, width):
    #     return 1 / np.cosh(t / width, dtype='complex')
    #
    # def gaussian(t, width):
    #     return np.exp(-0.5 * (np.power(t / width, 2)), dtype='complex')

    dt = t[1] - t[0]
    impulse_inds = np.arange(pulse_rep_t / dt * 0.5, pulse_rep_t / dt * (np.round((t[-1] - t[0]) / pulse_rep_t) + 0.5), pulse_rep_t / dt).astype('int')
    impulses = np.zeros_like(t).astype('complex')
    impulses[impulse_inds] = 1
    # if pulse_shape == 'gaussian':
    #     pulse = gaussian(t, pulse_width)
    # elif pulse_shape == 'sech':
    #     pulse = sech(t, pulse_width)
    pulse_train = numpy.fft.ifft(numpy.fft.fft(pulse, axis=0) * numpy.fft.fft(impulses, axis=0), axis=0)

    return pulse_train

