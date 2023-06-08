#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

T = 0.0001
sampling = int(1 / T)
t = np.arange(0, .3, T)

x = np.sin(np.pi  * 10 * t)
x += np.sin(np.pi * 100 * t)
x += np.sin(np.pi * 500 * t)
x += np.sin(np.pi * 1000 * t)
x += np.sin(np.pi * 1600 * t)

plt.plot(t, x)
# plt.show()
b, a = butter(4, [50, 150], fs=sampling, btype='bandstop')
x = lfilter(b, a, x)

b, a = butter(4, [350, 750], fs=sampling, btype='bandstop', analog=False)
x = lfilter(b, a, x)

b, a = butter(4, [900, 1500], fs=sampling, btype='bandstop', analog=False)
x = lfilter(b, a, x)

plt.plot(t, x)
plt.show()
