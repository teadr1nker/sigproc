#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import re
from mpl_toolkits import mplot3d
from scipy.signal import correlate, correlation_lags
from scipy.optimize import least_squares

speed = 1125    # Speed of sound
fs = 100000     # Sampling

# Speaker placement
speakers = np.array([[0, 0, 10],
                     [20, 0, 10],
                     [20, 20, 10],
                     [0, 20, 10]])

# Reading speaker data
f = open('Transmitter.txt', 'r')
text = f.read()
records = np.zeros((500, 4), dtype=np.float64) # Array of speaker signals
for i, row in enumerate(text.split('\n')[:-1]):
    records[:, i] = np.array(re.split('[ \t]+', row))[1:].astype(np.float64)

# Reading reciever data
f = open('Receiver.txt', 'r')
text = f.read()
received = np.array(re.split('[ \t]+', text))[1:].astype(np.float64)

# Getting distance from each speaker to microphone
# by calculating correlation lags
distance = np.zeros(4)
for i in range(4):
    cor = correlate(received, records[:,i])
    lags = correlation_lags(len(received), len(records[:,i]))
    distance[i] = lags[np.argmax(cor)] * speed / fs

# Minimizatiom function for least squares method
def f(x):
    position = np.zeros(4)
    for i in range(0, 4):
        position[i] = np.linalg.norm(speakers[i] - x)
    return position - distance

# Applying least squares method
ret = least_squares(f, [1, 1, 1])

# Show on 3d plot
fig = plt.figure()
ax = plt.axes(projection='3d')
x, y, z = speakers.T[0], speakers.T[1], speakers.T[2],
ax.scatter3D(x, y, z)
ax.scatter3D(ret.x[0], ret.x[1], ret.x[2])
plt.legend(['Speakers', f'Microphone {ret.x.round(2)}'])
plt.show()



