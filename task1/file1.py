#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

def updateState(state, st, sRot, T):
    return np.matrix([[state[0, 0] + T * st * np.cos(state[2, 0]) - (T * T * st * sRot * np.sin(state[2, 0])) / 2],
                      [state[1, 0] + T * st * np.sin(state[2, 0]) + (T * T * st * sRot * np.cos(state[2, 0])) / 2],
                      [state[2, 0] + T * sRot]])

def FJacobian(st, sRot, r, T):
    return np.matrix([[1, 0, -T*st*np.sin(r) - 0.5 * T **2 * st * sRot * np.cos(r)],
                      [0, 1, T*st*np.cos(r) - 0.5 * T ** 2 * st * sRot * np.sin(r)],
                      [0, 0, 1]])

# state estimate
m_k = np.matrix([[0],
                 [0],
                 [0]])
state = np.matrix([[0],
                   [0],
                   [0]])
# estimate covariance
P_k = np.matrix([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])
# noise covariance
R_k = np.matrix([[0.5, 0, 0],
                 [0, 0.5, 0],
                 [0, 0, 0.5]])
# noise mean
R_mean = np.array([0, 0, 0])

Wr = 3
Wl = 3
# base = 3
T = 1

points = np.array([0, 0, 0])        # state + noise
points_m = np.array([0, 0, 0])      # state denoised
pointsTrue = np.array([0, 0, 0])    # real state


for i in range(60):
    if i < 30:
        Ur = .3
        Ul = .2
    else:
        Ur = .2
        Ul = .3

    sr, sl = Wr * Ur, Wl * Ul

    st = (sr + sl) / 2
    sRot = (sr - sl) / 2

    state = updateState(state, st, sRot, T)
    # adding noise
    z_k = state + np.asmatrix(np.random.multivariate_normal(R_mean, R_k)).T

    # Predict
    m_k = updateState(m_k, st, sRot, T)         # Predicted state estimate
    Fx = FJacobian(st, sRot, m_k[2, 0], T)      # State transition model
    P_k = Fx * P_k * Fx.T                       # Predicted estimate covariance

    # Update
    S_k = P_k + R_k                             # Innovation  covariance
    K_k = P_k * np.linalg.inv(S_k)              # Optimal Kalman gain
    m_k = m_k + K_k * (z_k - m_k)               # Updated state estimate
    P_k = P_k - K_k * S_k * K_k.T               # Updated estimate covariance

    points = np.vstack([points, z_k.T])
    points_m = np.vstack([points_m, m_k.T])
    pointsTrue = np.vstack([pointsTrue, state.T])


plt.plot(pointsTrue[:, 0], pointsTrue[:, 1], marker="o")
plt.savefig("true")
plt.clf()

plt.plot(points[:, 0], points_m[:, 1], marker="o")
plt.savefig("raw")
plt.clf()

plt.plot(points_m[:, 0], points_m[:, 1], marker="o")
plt.savefig("filtered")
plt.clf()

