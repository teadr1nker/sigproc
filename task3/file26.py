#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby1, lfilter
import pandas as pd

# read signal
df = pd.read_csv('../task2/data/ecg.dat', delimiter=' ')
print(df.head(6))
t = df['time'].values
signal = df['signal'].values
n = df.size
T = t[2] - t[1]
sampling = int(1 / T)
print(f'size: {n}, sampling: {sampling}')
plt.plot(t[:4000], signal[:4000])

# filter signal
b, a = cheby1(10, 5, 30 , fs=sampling)
filtered = lfilter(b, a, signal)

plt.plot(t[:4000], filtered[:4000])
plt.show()
plt.clf()

# Calculating amplitude spectrum
fft1 = np.fft.fft(filtered)
frq = np.fft.fftfreq(len(filtered), T)
fft2 = np.fft.fft(signal)
plt.plot(frq, np.abs(fft2))
plt.plot(frq, np.abs(fft1))
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Filtered Signal Spectrum')
plt.show()
