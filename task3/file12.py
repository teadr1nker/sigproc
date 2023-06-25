#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

def bandpassFilter(ntaps, fs, bands):
    coef = 2 * np.pi / fs
    omega = np.arange(0, np.pi, np.pi / ntaps )
    freqResponse = np.zeros(ntaps, dtype=np.float64)

    # Finding frequency response
    for band in bands:
        start = coef * band[0]
        end = coef * band[1]
        freqResponse += np.where(np.logical_and(omega >= start, omega < end), 1, 0)

    nyq = (ntaps - 1)//2
    A = np.asmatrix(freqResponse).T
    FS = np.asmatrix(np.zeros((ntaps, nyq)))
    nyqRange = np.arange(nyq , 0, -1)

    # Filling FS matrix
    for i in range(ntaps - 1):
        FS[i, 0 : nyq] = 2 * np.sin(omega[i + 1] * nyqRange)

    # Calculating only left half impulse response
    h = np.linalg.inv(FS.T * FS) * FS.T * A
    h = np.asarray(h.T)[0]

    return np.concatenate([h, [0], -np.flip(h)])
