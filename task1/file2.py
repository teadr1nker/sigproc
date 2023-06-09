#!/usr/bin/python3
import numpy as np
import random
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# read files
sensorsData = []
with open('sensor_data_ekf.dat', 'r') as f:
    sdp = f.readlines()
    sensorsData = []

    for i in range(0, len(sdp)):
        st = sdp[i].split(" ")

        if st[0] == "ODOMETRY":
            s_d = []
            s_d.append([float(st[1]), float(st[2]), float(st[3])])
            sensorsData.append(s_d)
        elif st[0] == "SENSOR":
            s_d.append([int(st[1]), float(st[2]), float(st[3])])
        else:
            print("ERROR")

landmaksData = []
with open('landmarks.dat', 'r') as f:
    landmarks = f.readlines()
    for i in range(len(landmarks)):
        lm = landmarks[i].split(" ")
        l_s = [int(lm[0]), float(lm[1]), float(lm[2])]
        landmaksData.append(l_s)

###############################################################################
# Jacobian f
def get_Fx(theta, delta, k):
    return np.array([[1, 0, - delta[1] * math.sin(theta[k] + delta[0])],
                     [0, 1, delta[1] * math.cos(theta[k] + delta[0])],
                     [0, 0, 1]])


# Jacobian h
def get_Hx(delta_x, delta_y, q):
    H = np.zeros((3))
    H[0] = - delta_x / math.sqrt(q)
    H[1] = - delta_y / math.sqrt(q)

    return H

t = len(sensorsData)  # size of sample

###############################################################################
# Mean
m_x = [0 for i in range(t)]
m_y = [0 for i in range(t)]
m_r = [0 for i in range(t)]

# Inital means
m_x[0] = 0
m_y[0] = 0
m_r[0] = 0.1

# Covariation matrix
P = np.diag(np.full(3, 0.3))

# Noise covariation
Q = np.diag(np.full(3, 0.2))

# Extended Kalman Filter
for k in range(t - 1):
    delta = sensorsData[k][0]
    m_x[k + 1] = m_x[k] + delta[1] * math.cos(m_r[k] + delta[0])
    m_y[k + 1] = m_y[k] + delta[1] * math.sin(m_r[k] + delta[0])
    m_r[k + 1] = m_r[k] + delta[0] + delta[2]

    F = get_Fx(m_r, delta, k)
    F_P = F.dot(P)
    P = F_P.dot(np.transpose(F)) + Q

    # Correction
    n = len(sensorsData[k]) - 1  # Number of sensors
    H = np.zeros((n, 3))  # Jacobian h
    h = np.zeros((n))

    for i in range(1, n + 1):
        j = sensorsData[k][i][0] - 1

        delta_x = landmaksData[j][1] - m_x[k + 1]
        delta_y = landmaksData[j][2] - m_y[k + 1]

        H[i - 1, :] = get_Hx(delta_x, delta_y, delta_x * delta_x + delta_y * delta_y)
        h[i - 1] = math.sqrt(delta_x * delta_x + delta_y * delta_y)

    # S_k
    H_P = H.dot(P)
    R = np.diag(np.full(n, 0.2))  # Noise covariation matrix
    S = H_P.dot(np.transpose(H)) + R

    # K_k
    P_H = P.dot(np.transpose(H))
    K = P_H.dot(inv(S))

    # Get samples
    data = []
    for i in range(1, n + 1):
        data.append(sensorsData[k][i][1])

        y_k = np.array(data)

    # m_k
    m_k = np.array([m_x[k + 1], m_y[k + 1], m_r[k + 1]])
    m = m_k + K.dot(y_k - h)
    m_x[k + 1] = m[0]
    m_y[k + 1] = m[1]
    m_r[k + 1] = m[2]

    # P_k
    K_S = K.dot(S)
    P = P - K_S.dot(np.transpose(K))

EKF_x = m_x
EKF_y = m_y

###############################################################################
# Unscented Kalman Filter
n = 3

# Mean
m_x = np.zeros(t)
m_y = np.zeros(t)
m_r = np.zeros(t)

# Initial means
m_x[0] = 0
m_y[0] = 0
m_r[0] = 0.1

# Covariation matrix
P = np.zeros((3, 3))
P = np.diag(np.full(3, 0.5))

# Noise covariation
Q = np.diag(np.full(3, 0.2))

# Sigma points
X_x = np.zeros(2 * n + 1)
X_y = np.zeros(2 * n + 1)
X_r = np.zeros(2 * n + 1)

# Function values in points
Y = [0 for i in range(2 * n + 1)]

alpha = 1
beta = 0

for k in range(t - 1):
    l = alpha * alpha * (n + k) - n  #Lambda

    # Forecasting
    C = np.linalg.cholesky(P)  # Cholesky decomposition

    # Формирование сигма-точек
    X_x[0] = m_x[k]
    X_y[0] = m_y[k]
    X_r[0] = m_r[k]

    for i in range(1, n + 1):
        X_x[i] = m_x[k] + math.sqrt(n + l) * C[0][i - 1]
        X_y[i] = m_y[k] + math.sqrt(n + l) * C[1][i - 1]
        X_r[i] = m_r[k] + math.sqrt(n + l) * C[2][i - 1]

        X_x[i + n] = m_x[k] - math.sqrt(n + l) * C[0][i - 1]
        X_y[i + n] = m_y[k] - math.sqrt(n + l) * C[1][i - 1]
        X_r[i + n] = m_r[k] - math.sqrt(n + l) * C[2][i - 1]

    # Get values in sigma points
    delta = sensorsData[k][0]
    for i in range(2 * n + 1):
        X_x[i] = X_x[i] + delta[1] * math.cos(X_r[i] + delta[0])
        X_y[i] = X_y[i] + delta[1] * math.sin(X_r[i] + delta[0])
        X_r[i] = X_r[i] + delta[0] + delta[2]

    # m_k-
    w_m_0 = l / (n + l)
    w_m_i = 1 / (2 * (n + l))
    for i in range(2 * n + 1):
        if i == 0:
            m_x[k + 1] += w_m_0 * X_x[i]
            m_y[k + 1] += w_m_0 * X_y[i]
            m_r[k + 1] += w_m_0 * X_r[i]
        else:
            m_x[k + 1] += w_m_i * X_x[i]
            m_y[k + 1] += w_m_i * X_y[i]
            m_r[k + 1] += w_m_i * X_r[i]

    # P_k-
    w_c_0 = l / (n + l) + (1 - alpha * alpha + beta * beta)
    w_c_i = 1 / (2 * (n + l))

    m_k = np.array([[m_x[k + 1]], [m_y[k + 1]], [m_r[k + 1]]])
    P = np.zeros((3, 3))
    for i in range(2 * n + 1):
        X = np.array([[X_x[i]], [X_y[i]], [X_r[i]]])
        M = (X - m_k).dot(np.transpose(X - m_k))
        if i == 0:
            P += w_c_0 * M
        else:
            P += w_c_i * M

    P += Q

    # Correction
    C = np.linalg.cholesky(P)  # Cholesky decomposition

    # sigma points
    X_x[0] = m_x[k + 1]
    X_y[0] = m_y[k + 1]
    X_r[0] = m_r[k + 1]

    for i in range(1, n + 1):
        X_x[i] = m_x[k + 1] + math.sqrt(n + l) * C[0][i - 1]
        X_y[i] = m_y[k + 1] + math.sqrt(n + l) * C[1][i - 1]
        X_r[i] = m_r[k + 1] + math.sqrt(n + l) * C[2][i - 1]

        X_x[i + n] = m_x[k + 1] - math.sqrt(n + l) * C[0][i - 1]
        X_y[i + n] = m_y[k + 1] - math.sqrt(n + l) * C[1][i - 1]
        X_r[i + n] = m_r[k + 1] - math.sqrt(n + l) * C[2][i - 1]

    # Calculating sigma points
    m = len(sensorsData[k]) - 1  # Number of sensors
    for i in range(2 * n + 1):
        h = np.zeros((m, 1))
        for idx in range(1, m + 1):
            j = sensorsData[k][idx][0] - 1

            delta_x = landmaksData[j][1] - X_x[i]
            delta_y = landmaksData[j][2] - X_y[i]

            h[idx - 1][0] = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        Y[i] = h

    # mu_k
    mu_k = np.zeros((m, 1))
    for i in range(2 * n + 1):
        if i == 0:
            mu_k += w_m_0 * Y[i]
        else:
            mu_k += w_m_i * Y[i]

    # S_k
    R = np.diag(np.full(m, 0.2))  # Noise covariation matrix
    S_k = np.zeros((m, m))
    for i in range(2 * n + 1):
        M = (Y[i] - mu_k).dot(np.transpose(Y[i] - mu_k))
        if i == 0:
            S_k += w_c_0 * M
        else:
            S_k += w_c_i * M

    S_k += R

    # C_k
    C_k = np.zeros((3, m))
    m_k = np.array([[m_x[k + 1]], [m_y[k + 1]], [m_r[k + 1]]])
    for i in range(2 * n + 1):
        X = np.array([[X_x[i]], [X_y[i]], [X_r[i]]])
        M = (X - m_k).dot(np.transpose(Y[i] - mu_k))
        if i == 0:
            C_k += w_c_0 * M
        else:
            C_k += w_c_i * M

    K = C_k.dot(inv(S_k))  # K_k

    # Samples
    y_k = np.zeros((m, 1))
    for i in range(1, m + 1):
        y_k[i - 1] = sensorsData[k][i][1]

    # m_k
    m_k = m_k + K.dot(y_k - mu_k)

    m_x[k + 1] = m_k[0][0]
    m_y[k + 1] = m_k[1][0]
    m_r[k + 1] = m_k[2][0]

    # P_k
    K_S = K.dot(S_k)
    P = P - K_S.dot(np.transpose(K))

###############################################################################
# Plotting
# fig = plt.figure()
#
# # EKF
# ax1 = fig.add_subplot(2, 2, 1)
# ax1.plot(EKF_x, EKF_y, "-bo")
# ax1.set_title("EKF")
#
# # UKF
# ax2 = fig.add_subplot(2, 2, 2)
# ax2.plot(m_x, m_y, "-ro")
# ax2.set_title("UKF")
# plt.show()

plt.plot(EKF_x, EKF_y, "-bo")
plt.title('EKF')
plt.savefig('ekf.png')
plt.clf()
plt.plot(m_x, m_y, "-bo")
plt.title('UKF')
plt.savefig('ukf.png')
plt.clf()
