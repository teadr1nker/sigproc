#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby1, lfilter

T = 0.0001
sampling = int(1 / T)
t = np.arange(0, .3, T)

# create signal
x = np.sin(np.pi * 2  * 10 * t)
x += np.sin(np.pi * 2 * 100 * t)
x += np.sin(np.pi * 2 * 500 * t)
x += np.sin(np.pi * 2 * 1000 * t)
x += np.sin(np.pi * 2 * 1600 * t)

plt.plot(t, x)

# filter signal
b, a = cheby1(4, 4, [50, 150], fs=sampling, btype='bandstop')
filtered = lfilter(b, a, x)

b, a = cheby1(4, 4, [350, 750], fs=sampling, btype='bandstop', analog=False)
filtered = lfilter(b, a, filtered)

b, a = cheby1(4, 4, [900, 1500], fs=sampling, btype='bandstop', analog=False)
filtered = lfilter(b, a, filtered)

plt.plot(t, filtered)
plt.show()
plt.clf()

# Calculating amplitude spectrum
fft1 = np.fft.fft(filtered)
frq = np.fft.fftfreq(len(x), T)
fft2 = np.fft.fft(x)
plt.plot(frq, np.abs(fft2))
plt.plot(frq, np.abs(fft1), '-bo')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Filtered Signal Spectrum')
plt.show()

