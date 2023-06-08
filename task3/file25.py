#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import cheby1, lfilter, freqz

sampling, signal = wav.read('../task2/data/tune.wav')
print(f'sampling: {sampling}')

b, a = cheby1(10, 4, 6000, fs=sampling)
filtered = lfilter(b, a, signal).astype(np.int16)

wav.write('filtered.wav', sampling, filtered)

offset = sampling
size = sampling // 25
plt.plot(signal[offset:offset+size])
plt.plot(filtered[offset:offset+size])
plt.show()

