#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

# import autograd.numpy as np
from config.config import np

class Propagator(object):
    """Class for the state of an optical field within a system
    """

    def __init__(self, n_samples = None, window_t = None, central_wl = None):
        self._n_samples = n_samples
        self._window_t = window_t
        self._central_wl = central_wl

        self._central_frequency = self.speed_of_light / central_wl

        self._generate_time_frequency_arrays()

        self._state = np.zeros(self._t.shape, dtype='complex')
        return

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if type(state) is not np.ndarray:
            raise TypeError("Propagator state should be a numpy array")
        self._state = state
        return

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n_samples):
        self._n_samples = n_samples
        self._generate_time_frequency_arrays()
        return

    @property
    def window_t(self):
        return self._window_t

    @window_t.setter
    def window_t(self, window_t):
        if type(window_t) not in ['float', 'int']:
            raise ValueError("window_t should be a float value")
        self._window_t = window_t
        self._generate_time_frequency_arrays()
        return

    @property
    def central_wl(self):
        return self._central_wl

    @property
    def central_frequency(self):
        return self._central_frequency

    @property
    def speed_of_light(self):
        return 299792458

    @property
    def t(self):
        return self._t

    @property
    def f(self):
        return self._f

    @property
    def dt(self):
        return self._dt

    @property
    def df(self):
        return self._df

    def _generate_time_frequency_arrays(self):
        # temporal domain
        t, dt = np.linspace(-self._window_t / 2, self._window_t / 2, self._n_samples, retstep=True)  # time in s
        t = t.reshape((self._n_samples, 1))

        # frequency domain
        f, df = np.linspace(-self._n_samples / self._window_t / 2, self._n_samples / self._window_t / 2, self._n_samples,
                            retstep=True)  # frequency in Hz
        f = f.reshape((self.n_samples, 1))
        (self._t, self._f, self._dt, self._df) = (t, f, dt, df)
        return
