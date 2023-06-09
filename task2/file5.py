#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Read file
sampling, signal = wav.read('data/tune.wav')
n = signal.shape[0]
print(f'sampling: {sampling}, size: {n}')

# Forward Fourier transform
y = np.fft.fft(signal)

# Plot amplitude spectrum
amp = np.abs(y)
freq = np.fft.fftfreq(n, sampling) * sampling ** 2
plt.plot(freq, amp)
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.show()
plt.clf()

# Remove noise
yMod = np.array(y)
filt = np.array([0 if abs(f) >= 6000 else 1 for f in freq])
yMod *= filt

# Plot filtered amplitude spectrum
plt.plot(freq, np.abs(yMod))
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.show()

# Inverse Fourier transform
xMod = np.real(np.fft.ifft(yMod)).astype(np.int16)

# Write file
wav.write('filtered.wav', sampling, xMod)
