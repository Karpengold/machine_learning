import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import optimize


def add_params(X_original, p=0):
    X_copy = X_original.copy()
    for i in range(2, p + 1):
        X_copy[i] = X_copy[0] ** i
    return X_copy



def normalize(X_original):
    x_scaled = (X_original - train_means) / train_std
    df = pd.DataFrame(x_scaled)
    return df


def loss(theta, X, y, lmbd=0):
    units = np.full((len(X)), 1)  # единичный вектор
    return ((X @ theta - y) ** 2 @ units + lmbd * (np.sum(theta[1:] ** 2))) / (len(y))


def gradient(theta, X, y, lmbd=0):
    diff =   np.abs(lmbd*theta)
    diff[0] = 0
    return (2 / len(y)) *( X.T @ (X @ theta - y) + diff )


def learning_curves_chart(X_train, y_train, X_val, y_val, lambda_=0):
    m = len(y_train)
    train_err = np.zeros(m)
    val_err = np.zeros(m)
    theta = np.zeros(X_train.shape[1])
    for i in range(1, m):
        theta_bfgs = optimize.fmin_bfgs(
            loss,
            theta.flatten(),
            gradient,
            (X_train[0:i + 1, :], y_train[0:i + 1], lambda_)
        )
        train_err[i] = loss(theta_bfgs, X_train[0:i + 1, :], y_train[0:i + 1])
        val_err[i] = loss(theta_bfgs, X_val, y_val)
    plt.plot(range(2, m + 1), train_err[1:], c="r", linewidth=2)
    plt.plot(range(2, m + 1), val_err[1:], c="b", linewidth=2)
    plt.xlabel("x count", fontsize=14)
    plt.ylabel("error", fontsize=14)
    plt.legend(["training", "validation"])
    plt.axis([2, m, 0, max(max(val_err), max(train_err))])
    plt.grid()
    plt.show()


# 1
dataset = sio.loadmat("ex3data1.mat")

x_train = pd.DataFrame(dataset["X"])
x_val = pd.DataFrame(dataset["Xval"])
x_test = pd.DataFrame(dataset["Xtest"])

y_train = dataset["y"].squeeze()
y_val = dataset["yval"].squeeze()
y_test = dataset["ytest"].squeeze()


#2

fig, ax = plt.subplots()
ax.scatter(x_train, y_train)
plt.xlabel("water level", fontsize=14)
plt.ylabel("out", fontsize=14)
plt.show()

# 3
x_train_ones = x_train.copy()
x_train_ones.insert(0, 'Ones', 1)
theta = np.zeros(x_train_ones.shape[1])
print('Floss: ', loss(theta, x_train_ones, y_train, 0))

# 4
print('Gradient: ', gradient(theta, x_train_ones, y_train, 0))

# 5

theta_bfgs = optimize.fmin_bfgs(
    loss,
    theta.flatten(),
    gradient,
    (x_train_ones.values, y_train, 0)
)
print('Theta: ', theta_bfgs)
print('Loss: ', loss(theta_bfgs, x_train_ones, y_train, 0))

h = np.dot(x_train_ones, theta_bfgs)
fig, ax = plt.subplots()
plt.title('1')
ax.scatter(x_train, y_train)
ax.plot(x_train, h, linewidth=2, color='red')
plt.show()

# 6
x_val_ones = x_val.copy()
x_val_ones.insert(0, 'Ones', 1)
learning_curves_chart(x_train_ones.values, y_train, x_val_ones.values, y_val, 0)

# 7,8

x_poly =add_params(x_train, 8)
train_means = x_poly.mean(axis=0)
train_std = np.std(x_poly, axis=0, ddof=1)
x_scaled = normalize(x_poly)
x_scaled.insert(0, 'Ones', 1)



# 9
theta = np.zeros(x_scaled.shape[1])
theta_bfgs_scaled =  theta_bfgs = optimize.fmin_bfgs(
    loss,
    theta.flatten(),
    gradient,
    (x_scaled.values, y_train, 0)
)
print(loss(theta_bfgs_scaled, x_scaled, y_train, 0))

# 10

x = pd.DataFrame(np.linspace(min(x_train.values) - 5, max(x_train.values) + 5, 1000))

x_polynom = add_params(x, 8)
x_polynom = normalize(x_polynom)
x_polynom.insert(0, 'Ones', 1)

fig, ax = plt.subplots()
plt.scatter(x_train.values, y_train, color='red')
plt.plot(x, x_polynom @ theta_bfgs_scaled, linewidth=2)
plt.xlabel("water level", fontsize=14)
plt.ylabel("out", fontsize=14)
plt.show()



val = normalize(add_params(x_val, 8))
val.insert(0, 'Ones', 1)
learning_curves_chart(x_scaled.values, y_train, val.values, y_val, 0)

# 11
theta_bfgs_scaled = optimize.fmin_bfgs(
    loss,
    theta.flatten(),
    gradient,
    (x_scaled.values, y_train, 1)
)

plt.scatter(x_train.values, y_train, color='red')
plt.plot(x, np.dot(x_polynom, theta_bfgs_scaled), linewidth=2)
plt.xlabel("water level", fontsize=14)
plt.ylabel("out", fontsize=14)
plt.show()

learning_curves_chart(x_scaled.values, y_train, val.values, y_val, 1)

theta_bfgs_scaled = optimize.fmin_bfgs(
    loss,
    theta.flatten(),
    gradient,
    (x_scaled.values, y_train, 100)
)
plt.scatter(x_train.values, y_train, color='red')
plt.plot(x, np.dot(x_polynom, theta_bfgs_scaled), linewidth=2)
plt.xlabel("water level", fontsize=14)
plt.ylabel("out", fontsize=14)
plt.show()
learning_curves_chart(x_scaled.values, y_train, val.values, y_val, 100)

# 12
lambda_values = np.linspace(0,10, 1000)
val_err = []
for lamb in lambda_values:
    theta_bfgs_scaled = optimize.fmin_bfgs(
        loss,
        theta.flatten(),
        gradient,
        (x_scaled.values, y_train, lamb)
    )
    val_err.append(loss(theta_bfgs_scaled, val, y_val))
plt.plot(lambda_values, val_err, c="b", linewidth=2)
plt.grid()
plt.xlabel("lambda", fontsize=14)
plt.ylabel("error", fontsize=14)
plt.show()
print(lambda_values[np.argmin(val_err)])


#13
theta = np.zeros(x_scaled.shape[1])
x_test = normalize(add_params(x_test, 8))
x_test.insert(0, "Ones", 1)
theta_bfgs_scaled = optimize.fmin_bfgs(
        loss,
        theta.flatten(),
        gradient,
        (x_scaled.values, y_train, 2.97)
)
print(loss(theta_bfgs_scaled, x_test, y_test))