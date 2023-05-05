#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

def updateState(state, st, sRot, T):
    return np.matrix([[state[0,0] + T * st * np.cos(state[2,0]) - (T * T * st * sRot * np.sin(state[2,0])) / 2],
                      [state[1,0] + T * st * np.sin(state[2,0]) + (T * T * st * sRot * np.cos(state[2,0])) / 2],
                      [state[2,0] + T * sRot]])

def FJacobian(st, sRot, r, T):
    return np.matrix([[1, 0, -T*st*np.sin(r) - 0.5*T**2*st*sRot*np.cos(r)],
                      [0, 1, T*st*np.cos(r) - 0.5*T**2*st*sRot*np.sin(r)],
                      [0, 0, 1]])


mless_k = np.matrix([[0], [0], [0]])
m_k = np.matrix([[0], [0], [0]])
state = np.matrix([[0], [0], [0]])

Pless_k = np.matrix([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
P_k = np.matrix([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])
K_k = np.matrix([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])

R_k = np.matrix([[0.5, 0, 0],
                 [0, 0.5, 0],
                 [0, 0, 0.5]])

R_mean = np.array([0, 0, 0])

Wr = 3
Wl = 3
# base = 3

T = 1

points = np.array([0, 0, 0])
points_m = np.array([0, 0, 0])
pointsTrue = np.array([0, 0, 0])


for i in range(60):
    if i < 30:
        Ur = .3
        Ul = .2
    else:
        Ur = .2
        Ul = .3

    sr = Wr * Ur
    sl = Wl * Ul

    st = (sr + sl) / 2
    sRot = (sr - sl) / 2

    state = updateState(state, st, sRot, T)
    # adding noise
    z_k = state + np.asmatrix(np.random.multivariate_normal(R_mean, R_k)).T

    # Forecast
    mless_k = updateState(m_k, st, sRot, T)
    Fx = FJacobian(st, sRot, m_k[2,0], T)
    Pless_k = Fx * Pless_k * Fx.T

    # Correction
    S_k = Pless_k + R_k
    K_k = Pless_k * np.linalg.inv(S_k)
    m_k = mless_k + K_k * (z_k - mless_k)
    Pless_k = Pless_k - K_k * S_k * K_k.T

    points = np.vstack([points, z_k.T])
    points_m = np.vstack([points_m, m_k.T])
    pointsTrue = np.vstack([pointsTrue, state.T])


plt.plot(pointsTrue[:,0], pointsTrue[:,1], marker="o", linestyle="None")
plt.savefig("true")
plt.clf()

plt.plot(points[:,0], points_m[:,1], marker="o", linestyle="None")
plt.savefig("raw")
plt.clf()

plt.plot(points_m[:,0], points_m[:,1], marker="o", linestyle="None")
plt.savefig("filtered")
plt.clf()

