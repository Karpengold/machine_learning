import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.patches as mpatches
offset = 125


def predict_number(theta_arr, x):

    return np.dot(x, np.array(theta_arr).T)

def sig(z):
    return 1 / (1 + np.exp(-z))


def loss(w, X, y):
    h = sig(np.dot(X, w))
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def loss_L2(theta, X,y, k=0.0):
    h = sig(np.dot(X, theta))
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + (k / (2 * m)) * np.sum(theta[1:]**2)

def gradient(w, X, y):
    h = sig(np.dot(X, w))
    return np.dot(X.T, (h - y)) / len(y)
def gradient_regularized(theta, X, y, k):
    m = len(y)
    grad = gradient(theta, X ,y)
    grad[1:] = grad[1:]+ (k / m) * theta[1:]
    return grad
def gradient_regularized1(theta, X, y, k=0):
    m = len(y)
    grad = (1 / m) * np.dot(X.T, (sig(np.dot(X, theta)) - y))
    grad[1:] = grad[1:] + ((2*k) / m) * theta[1:]
    return grad

def normalize(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    df = pd.DataFrame(x_scaled)
    return df


def gradient_descent(X, y, w, learning_rate=0.0001, k=0.005, steps=105000):
    t = 1
    next_w = w - k * gradient(w, X, y)
    while np.linalg.norm(w - next_w) > learning_rate and t < steps:
        w = next_w
        next_w = w - k * gradient(w, X, y)
        t += 1
    return next_w




#1
data = pd.read_csv('ex2data1.txt', header=None)
num_columns = data.shape[1]
X = data.iloc[:, 0:num_columns - 1]
y = data[num_columns - 1]

#2
false_indexes = y == 0
true_indexes = y == 1
fail = plt.scatter(X[false_indexes][0].values, X[false_indexes][1].values)
ok = plt.scatter(X[true_indexes][0].values, X[true_indexes][1].values)
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.legend((ok, fail), ('Passed', 'Not passed'))
plt.show()

#3
X.insert(0, 'Ones', 1)
theta = gradient_descent(X.values, y.values, np.array([0, 0, 0]))
print(theta)
print(loss(theta, X, y))
#4

temp = optimize.minimize(loss, np.array([0, 0, 0]), (X, y), method='Nelder-Mead')
print(temp.x)
print(loss(temp.x, X, y))
theta_optimized = optimize.fmin_bfgs(
    loss,
    np.array([0, 0, 0]),
    gradient,
    (X, y)
)
print(theta_optimized)
print(loss(theta_optimized, X, y))

#6
x1 = np.array([np.min(X[0]), np.max(X[1])])
x2 = (-1 / theta_optimized[2]) * (x1 * theta_optimized[1]) + offset
#plt.plot(x1, x2)
#plt.show()



#7
data = pd.read_csv('ex2data2.txt', header=None)
num_columns = data.shape[1]
X = data.iloc[:, 0:num_columns - 1]
y = data[num_columns - 1]


#8

ok = y == 1
fail = y != 1
psd = plt.scatter(X[ok][0].values, X[ok][1].values)
not_psd = plt.scatter(X[fail][0].values, X[fail][1].values)
plt.xlabel('Test 1')
plt.ylabel('Test 2')
plt.legend((psd, not_psd), ('Passed', 'Failed'))



#9
def polynom_combs(x1, x2):
    res = []
    for i in range(0,7):
        for j in range(0,7):
            if i + j <= 6:
                res.append((x1**i)*(x2**j))
    return  res

values = X.values
X_combs = []

for i in range (0, len(values)):
    X_combs.append(polynom_combs(values[i][0], values[i][1]))
print(X_combs)

#10

X_combs = pd.DataFrame(X_combs)
X_combs.insert(0, 'Ones', 1)

(m, n) = X_combs.shape
theta = np.zeros((n , 1))
X = X_combs

theta_tnc = optimize.fmin_tnc(
    func=loss_L2,
    x0=theta.flatten(),
    fprime=gradient_regularized,
    args=(X, y,  0.1)
)
print('TNC theta: ', theta_tnc[0])
print('TNC loss:' ,loss(theta_tnc[0], X, y))

#11
theta_nm = optimize.minimize(loss_L2, theta.flatten(), (X, y, 0.1), method='Nelder-Mead')
print('Nelder-Mead theta: ', theta_nm.x)
print('Nelder-Mead loss:' ,loss(theta_nm.x, X, y))

theta_bfgs = optimize.fmin_bfgs(
    loss_L2,
    theta.flatten(),
    gradient_regularized,
    (X, y, 0.01)
)
print('Brovden Fletcher Goldfarb Shanno theta: ', theta_bfgs)
print('Brovden Fletcher Goldfarb Shanno loss:' ,loss(theta_bfgs, X, y))


#12
print('Predictions: ', sig(np.dot(X.values[0].T, theta_bfgs)))

#13, 14
theta_bfgs2 = optimize.fmin_bfgs(
    loss_L2,
    theta.flatten(),
    gradient_regularized,
    (X, y, 0.001)
)
theta_bfgs3 = optimize.fmin_bfgs(
    loss_L2,
    theta.flatten(),
    gradient_regularized,
    (X, y, 0.1)
)
xd = np.linspace(-1, 1, 50)
yd = np.linspace(-1, 1, 50)
z1 = np.zeros((len(xd), len(yd)))
z2 = np.zeros((len(xd), len(yd)))
z3 = np.zeros((len(xd), len(yd)))
for i in range(len(xd)):
    for j in range(len(yd)):
        dots = [1]
        dots.extend(polynom_combs(xd[i], yd[j]))
        z1[i, j] = sig(np.dot(np.array(dots).T, theta_bfgs))
for i in range(len(xd)):
    for j in range(len(yd)):
        dots = [1]
        dots.extend(polynom_combs(xd[i], yd[j]))
        z2[i, j] = sig(np.dot(np.array(dots).T, theta_bfgs2))
for i in range(len(xd)):
    for j in range(len(yd)):
        dots = [1]
        dots.extend(polynom_combs(xd[i], yd[j]))
        z3[i, j] = sig(np.dot(np.array(dots).T, theta_bfgs3))

mask = y.values.flatten() == 1
X = data.iloc[:, :-1]
passed = plt.scatter(X[mask][0], X[mask][1])
failed = plt.scatter(X[~mask][0], X[~mask][1])
plt.contour(xd, yd, z1, 0, colors='red')
plt.contour(xd, yd, z2, 0, colors= 'blue')
plt.contour(xd, yd, z3, 0, colors= 'orange')
blue_patch = mpatches.Patch(color='blue', label='lambda= 0.001')
red_patch = mpatches.Patch(color='red', label='lambda = 0.01')
orange_patch = mpatches.Patch(color='orange', label='lambda= 0.1')
plt.legend(handles=[blue_patch, red_patch, orange_patch])

plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')

plt.show()


#15
data = sio.loadmat('ex2data3.mat')
X = data.get('X')
y = data.get('y')

# 16
images = {}

for i in range(len(y)):
    images[y[i][0]] = i
keys = images.keys()

fig, axis = plt.subplots(1, 10)

for j in range(len(keys)):
    axis[j].imshow(X[images.get(list(keys)[j]), :].reshape(20, 20, order="F"), cmap="hot")
    axis[j].axis("off")

#plt.show()

#17
X = pd.DataFrame(X)
X.insert(0, 'Ones', 1)

(m, n) = X.shape
lmbda = 0.1
k = 10
theta = np.zeros((n, 1))  # initial parameters
print("F_loss ", loss(theta, X, y))
print("Gradient F_loss ", gradient(theta, X, y))

#18
print("Loss L2 ", loss_L2(theta, X, y, 0.01))
print("Gradient F_loss ", gradient_regularized(theta, X, y, 0.01))

#19
theta_arr = []*10

for i in range (0, 10):
    print(i)
    digit_class = i if i else 10
    theta_temp = optimize.fmin_bfgs(
        loss_L2,
        theta.flatten(),
        gradient_regularized,
        (X, (y == digit_class).flatten().astype(np.int), 0.1)
    )
    theta_arr.append(theta_temp)

#20
print('predict: ',  np.argmax(predict_number(X.values[0], theta_arr)), "Real: ", y[0][0])


#21
success = 0
predicted = predict_number(X.values, theta_arr).T
for i in range(len(predicted)):
    p = np.argmax(predicted[i])
    if  p == 0:
        p = 10
    if  p == y[i][0]:
        success += 1
print('Acc: ', success/len(predicted))
