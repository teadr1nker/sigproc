#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import cheby1, lfilter

# read file
sampling, signal = wav.read('../task2/data/tune.wav')
print(f'sampling: {sampling}')

# filter signal
b, a = cheby1(10, 4, 6000, fs=sampling)
filtered = lfilter(b, a, signal).astype(np.int16)

wav.write('filtered.wav', sampling, filtered)

# plot difference
offset = sampling
size = sampling // 25
plt.plot(signal[offset:offset+size])
plt.plot(filtered[offset:offset+size])
plt.show()
plt.clf()

# Calculating amplitude spectrum
frq = np.fft.fftfreq(len(filtered), 1/sampling)
fft1 = np.fft.fft(filtered)
fft2 = np.fft.fft(signal)
plt.plot(frq, np.abs(fft2))
plt.plot(frq, np.abs(fft1))
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Filtered Signal Spectrum')
plt.show()
