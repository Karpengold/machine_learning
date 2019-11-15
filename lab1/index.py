import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import preprocessing
import time

from sklearn.linear_model import LinearRegression


import numpy as np



def loss(X, y, w):
    units = np.full((len(X)), 1)  # единичный вектор
    return ((X @ w - y) ** 2 @ units) / len(X)


def gradient(X, y, w):
    n = len(w)
    grad = [0.0] * n
    l = len(X)
    for k in range(0, n):
        for i in range(0, l):
            temp = 0
            for m in range(0, n):
                temp += w[m] * X[i][m]
            temp -= y[i]
            grad[k] += temp * X[i][k]
        grad[k] = (grad[k] * 2) / l

    return grad


def gradient_descent(X, y, w, learning_rate=0.0001, k=0.01, steps=10000):
    n = len(w)
    t = 1

    grad = gradient(X, y, w)
    next_w = [0.0] * n
    for i in range(0, n):
        next_w[i] = w[i] - k  * grad[i]

    while (t < steps):
        w = next_w
        grad = gradient(X, y, w)
        for i in range(0, n):
            next_w[i] = w[i] - k * grad[i]
        t += 1

    return next_w


def gradient_vectorized(X, y, w):
    return (2 / len(X)) * X.T @ (X @ w - y)


def gradient_descent_vectorized(X, y, w, learning_rate=0.0001, k=0.1, steps=10000):
    t = 1
    next_w = w - k * gradient_vectorized(X, y, w)
    while np.linalg.norm(w - next_w) > learning_rate and t < steps:
        w = next_w
        next_w = w - k * gradient_vectorized(X, y, w)
        t += 1
    return next_w


def feature_normalization(X):
    X = X.T
    for i in range(1, len(X)):
        mu = np.mean(X[i])
        s = np.std(X[i], ddof=1)  # TODO: learn more about ddof
        X[i] = (X[i] - mu) / s
    return X.T


def model_plot(X, w):
    calculated_y = X @ w
    data.plot(kind='scatter', x=0, y=1)
    plt.plot(data[0].values, calculated_y)
    plt.show()


def loss_plot(X, y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    u = np.arange(-5, 5, 0.1)
    v = np.arange(-5, 5, 0.1)
    z = np.zeros((len(u), len(v)))
    for i in range(0, len(u)):
        for j in range(0, len(v)):
            z[i][j] = loss(X, y, [u[i], v[j]])

    u, v = np.meshgrid(u, v)

    # Plot the surface.
    surf = ax.plot_surface(u, v, z,
                           linewidth=0, antialiased=False)
    plt.show()


def lsm(X, y):
    return ((np.linalg.inv((X.T.dot(X)))).dot(X.T)) @ y


def loss_contour_plot(X, y):
    fig, ax = plt.subplots()

    u = np.arange(-20, 20, 0.2)
    v = np.arange(-10, 10, 0.1)
    z = np.zeros((len(u), len(v)))
    for i in range(0, len(u)):
        for j in range(0, len(v)):
            z[i][j] = loss(X, y, [u[i], v[j]])

    u, v = np.meshgrid(u, v)

    # Plot the contour plot.
    CS = ax.contour(u, v, z, levels=[20, 50, 125, 300, 750, 1800, 4500, 11250, 28000])
    # ax.clabel(CS, inline=1, fontsize=10)
    plt.show()

# 1
data = pd.read_csv('ex1data1.txt', header=None)
num_columns = data.shape[1]

panda_X = data.iloc[:, 0:num_columns - 1]  # [ slice_of_rows, slice_of_columns ]
panda_X.insert(0, 'Ones', 1)
X = panda_X.values
y = data[num_columns - 1].values

#2
data.plot.scatter(x=0, y =1)
plt.show()

#3
print(loss(X,y, [0,0]))

#4
theta = gradient_descent(X,y,[0,0])
print(theta)
model_plot(X, theta)


#5
loss_plot(X,y)
loss_contour_plot(X,y)


#6
data = pd.read_csv('ex1data2.txt', header=None)
num_columns = data.shape[1]
y = data[num_columns - 1].values


#7
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data.iloc[:, 0:num_columns - 1])
df = pd.DataFrame(x_scaled)
df.insert(0, 'Ones', 1)
x_scaled = df.values
print(x_scaled)

#8
start1 = time.time()
theta = gradient_descent_vectorized(x_scaled,y,np.array([0,0,0]))
time1 = time.time() - start1
print(theta)
print(loss(x_scaled, y, theta))


#9
start2 = time.time()
theta = gradient_descent(x_scaled,y, np.array([0,0,0]))
time2 = time.time() - start2
print(time1, time2)

#10
print('lsm')
theta = lsm(x_scaled,y)
print(theta)
print(loss(x_scaled, y, theta))
