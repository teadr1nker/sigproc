#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/ecg.dat', delimiter=' ')
print(df.head(6))
t = df['time'].values
signal = df['signal'].values
n = df.size
T = t[2] - t[1]
sampling = 1 // T
print(f'size: {n}, period: {T}')
plt.plot(t[:4000], signal[:4000])
plt.show()

# Forward Fourier transform
y = np.fft.fft(signal)

# Plot amplitude spectrum
amp = np.abs(y)
freq = np.fft.fftfreq(n // 2, int(1 / T)) * sampling ** 2
plt.plot(freq, amp)
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.show()
plt.clf()

# Remove noise
noiseFilter = np.array([0 if abs(f) >= 45 else 1 for f in freq])

yMod = noiseFilter * y


fraction = 2
yMod[(n//fraction):-(n//fraction)] = 0

plt.plot(freq, np.abs(yMod))
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.show()
plt.clf()

xMod = np.real(np.fft.ifft(yMod))
plt.plot(t[:4000], xMod[:4000])
plt.show()
