#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

def interpolate(t, X, T):
    res = 0.
    for k, x in enumerate(X):
        res += x * np.sinc((np.pi / T) * (t - (k * T)))
    return res

W = 5

T1 = 0.01
t1 = np.arange(0, 2, T1)
x1 = np.sin(np.pi * W * t1)

plt.plot(t1, x1)

T2 = 1 / 15
t2 = np.arange(0, 2, T2)
x2 = np.sin(np.pi * W * t2)

plt.plot(t2, x2, linestyle='', marker='o')

# size = len(t2)
# x2 = x2[:size//2]

x3 = np.array([interpolate(t, x2, T2) for t in t1])
plt.plot(t1, x3)
plt.show()
