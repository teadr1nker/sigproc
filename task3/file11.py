#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firls, freqz
from file12 import bandpassFilter

fs = 8000
ntaps = 501
bands = [0, 50, 50, 150,
         150, 350,
         350, 750, 750, 900,
         900, 1500, 1500, fs//2]
desired = (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0)

# Bultin function
taps = firls(ntaps, bands, desired, fs=fs)
w, h = freqz(taps)
plt.plot((.5 * fs / np.pi) * w, abs(h))

# Implemented function
bands = [[50, 150],
         [350, 750],
         [900, 1500]]
taps = bandpassFilter(ntaps, fs, bands)
w, h = freqz(taps)
plt.plot((.5 * fs / np.pi) * w, abs(h))

# Compare on plot
plt.legend(['scipy.signal.firls', 'implemented bandpass filter'])
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()

