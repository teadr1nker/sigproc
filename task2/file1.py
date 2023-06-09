#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

def interpolate(t, X, T):
    res = 0.
    for k, x in enumerate(X):
        res += x * np.sinc((np.pi / T) * (t - (k * T)))
    return res

W = 3

T1 = 0.01
t1 = np.arange(0, 3, T1)
x1 = np.sin(np.pi * 2 * W * t1)

plt.plot(t1, x1)

W2 = 10
T2 = 1 / W2
t2 = np.arange(0, 3, T2)
x2 = np.sin(np.pi * 2 * W * t2)

plt.plot(t2, x2, linestyle='', marker='o')
x3 = 0
for k, x in enumerate(x2):
    x3 += x * np.sinc(W2 * (t1 - k / W2))

plt.plot(t1, x3)
plt.show()
