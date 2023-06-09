#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
def X1(t):
    return np.exp(-(t ** 2))

def X2(T):
    return [np.cos((np.pi * t) / 2) if np.abs(t) <=1 else 0 for t in T]

def ampSpec(f):
    T = .01
    t = np.arange(0, 5, T)
    x = f(t)
    # plt.plot(t, x)
    # plt.show()
    # plt.clf()

    xFFT = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x), 100)
    freq = np.fft.fftshift(freq)
    amp = np.abs(xFFT)
    plt.plot(freq, amp)
    plt.show()

ampSpec(X1)
ampSpec(X2)
