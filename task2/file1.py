#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

W = 3

# create signal
T1 = 0.01
t1 = np.arange(0, 3, T1)
x1 = np.sin(np.pi * 2 * W * t1)

plt.plot(t1, x1)

# sample signal
W2 = 7
T2 = 1 / W2
t2 = np.arange(0, 3, T2)
x2 = np.sin(np.pi * 2 * W * t2)

plt.plot(t2, x2, linestyle='', marker='o')

# interpolation
x3 = 0
for k, x in enumerate(x2):
    x3 += x * np.sinc(W2 * (t1 - k / W2))

plt.plot(t1, x3)
plt.show()
