#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

timeStep = .01
sigFreq = 2
disFreq = 5

t = np.arange(0, 10, timeStep)
# print(t)
oSig = np.sin(t * np.pi * sigFreq)
plt.plot(t, oSig)

t2 = np.arange(0, 10, 1 / disFreq)
dSig = np.sin(t2 * np.pi * sigFreq)
plt.plot(t2, dSig)

# plt.show()
plt.clf()

sigFft = fft.fft(dSig)
amp = np.abs(sigFft)
power = amp ** 2
ang = np.angle(sigFft)
sFreq = fft.fftfreq(len(dSig), d = 1 / disFreq)

ampFreq = np.array([amp, sFreq])
peakFreq = ampFreq[1, ampFreq[0, :].argmax()]
print(peakFreq)
plt.plot(amp)
plt.show()
