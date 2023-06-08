#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheb1ord, cheby1, lfilter
import pandas as pd

df = pd.read_csv('../task2/data/ecg.dat', delimiter=' ')
print(df.head(6))
t = df['time'].values
signal = df['signal'].values
n = df.size
T = t[2] - t[1]
sampling = int(1 / T)
print(f'size: {n}, sampling: {sampling}')
plt.plot(t[:4000], signal[:4000])

b, a = cheby1(10, 5, 30 , fs=sampling)
filtered = lfilter(b, a, signal)

plt.plot(t[:4000], filtered[:4000])
plt.show()
