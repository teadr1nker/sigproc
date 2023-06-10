#!/usr/bin/python3
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

def desingFilter(fs, freqs):
    num_taps = 100  # Filter len

    # filter matrix creation
    A = np.zeros((num_taps, len(freqs)))
    b = np.zeros((num_taps, 1))

    for i, range in enumerate(freqs):
        start, end = range

        # indexes for filtering
        start_index = int(start * num_taps / fs)
        end_index = int(end * num_taps / fs)

        # Calculating vectors
        A[start_index:end_index, i] = 1
        b[start_index:end_index] = 1

    # Solving equation system
    coeffs, _, _, _ = lstsq(A, b)

    return coeffs.flatten()

T = 0.0001
sampling = int(1 / T)
t = np.arange(0, .3, T)

x = np.sin(np.pi * 2  * 10 * t)
x += np.sin(np.pi * 2 * 100 * t)
x += np.sin(np.pi * 2 * 500 * t)
x += np.sin(np.pi * 2 * 1000 * t)
x += np.sin(np.pi * 2 * 1600 * t)

freqs = [(50, 150), (350, 750), (900, 1500)]

# Calculating filter
filt = desingFilter(sampling, freqs)
print("Filter coefficients:", filt)

# Filtering
filtered = np.convolve(x, filt, mode='same')
filtered = lfilter(b, a, x)
plt.plot(t, filtered)
plt.plot(t, x)
plt.show()

# Calculating amplitude spectrum
fft1 = np.fft.fft(filtered)
frq = np.fft.fftfreq(len(x), T)
fft2 = np.fft.fft(x)
plt.plot(frq, np.abs(fft2))
plt.plot(frq, np.abs(fft1))
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Filtered Signal Spectrum')
plt.show()
