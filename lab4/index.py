import scipy.io as sio
import numpy as np
import pandas as pd
from scipy import optimize as opt
import matplotlib.pyplot as plt

def accuracy(y_pred, y_true):
    error = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i][0]:
            error += 1
    return 1 - error / len(y_pred)

def checkGradient(nn_initial_params,nn_backprop_Params,input_layer_size, hidden_layer_size, num_labels,myX,myy,mylambda=0., e= 0.0001):

    flattened = nn_initial_params
    flattenedDs = nn_backprop_Params
    n_elems = len(flattened)
    #Pick ten random elements, compute numerical gradient, compare to respective D's
    for i in range(10):
        x = int(np.random.rand()*n_elems)
        epsvec = np.zeros((n_elems,1))
        epsvec[x] = e

        cost_high = loss(flattened + epsvec.flatten(),input_layer_size, hidden_layer_size, num_labels,myX,myy,mylambda)
        cost_low  = loss(flattened - epsvec.flatten(),input_layer_size, hidden_layer_size, num_labels,myX,myy,mylambda)
        mygrad = (cost_high - cost_low) / float(2*e)
        print("Element: {0}. Numerical Gradient = {1:.9f}. BackProp Gradient = {2:.9f}.".format(x,mygrad,flattenedDs[x]))


def sigmoid(z):
    return 1/(1+np.exp(-z))

def hx(theta1, theta2, X):
    m = len(y)
    ones = np.ones((m, 1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(np.dot(a1, theta1.T))
    a2 = np.hstack((ones, a2))
    h = sigmoid(np.dot(a2, theta2.T))

    return h

def dsig(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))

def gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    initial_theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                                (hidden_layer_size, input_layer_size + 1), 'F')
    initial_theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                                (num_labels, hidden_layer_size + 1), 'F')
    y_d = pd.get_dummies(y.flatten())
    delta1 = np.zeros(initial_theta1.shape)
    delta2 = np.zeros(initial_theta2.shape)
    m = len(y)

    ones = np.ones((X.shape[0], 1))
    a1 = np.hstack((ones, X))
    z2 = np.dot(a1, initial_theta1.T)
    a2 = np.hstack((ones, sigmoid(z2)))
    z3 = np.dot(a2, initial_theta2.T)
    a3 = sigmoid(z3)

    d3 = a3 - y_d
    z2 = np.hstack((ones, z2))
    d2 = np.multiply(np.dot(initial_theta2.T,d3.T), dsig(z2).T)
    delta1 = delta1 + np.dot(d2[1:, :], a1)
    delta2 = delta2 + np.dot(d3.T, a2)

    delta1 /= m
    delta2 /= m
    delta1[:, 1:] = delta1[:, 1:] + initial_theta1[:, 1:] * lmbda / m
    delta2[:, 1:] = delta2[:, 1:] + initial_theta2[:, 1:] * lmbda / m

    return np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))

def loss(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    m = len(y)
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), 'F')
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1),'F')
    y = pd.get_dummies(y.flatten())
    h = hx(theta1, theta2, X)
    temp1 = np.multiply(y, np.log(h))
    temp2 = np.multiply(1 - y, np.log(1 - h))
    temp3 = np.sum(temp1 + temp2)

    sum1 = np.sum(np.sum(np.power(theta1[:, 1:], 2), axis=1))
    sum2 = np.sum(np.sum(np.power(theta2[:, 1:], 2), axis=1))

    return np.sum(temp3 / (-m)) + (sum1 + sum2) * lmbda / (2 * m)



def randInitializeWeights(L_in, L_out):
    epsilon = 0.12
    return np.random.rand(L_out, L_in+1) * 2 * epsilon - epsilon

def predict(theta1, theta2, X):
    return np.argmax(hx(theta1,theta2,X), axis = 1) + 1


#1
data = sio.loadmat('ex4data1.mat')
X = data.get('X')
y = data.get('y')

#2
weights = sio.loadmat('ex4weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']

nn_params = np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F')))    #unroll parameters
# neural network hyperparameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lmbda = 1


initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
# unrolling parameters into a single column vector
nn_initial_params = np.hstack((initial_theta1.ravel(order='F'), initial_theta2.ravel(order='F')))
print(gradient(nn_initial_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0))

#3
print(hx(theta1,theta2,X))
y_pred = predict(theta1, theta2, X)
print('Accuracy: ', accuracy(y_pred, y))

#5
y_one_hot = pd.get_dummies(y.flatten())
# 6
loss0 = loss(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
print("Loss unregularized:", loss0)

# 7

loss1 = loss(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
print("Loss regularized: ", loss1)

#8
print('Sigmoid grad: ', dsig(2))

#9

initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
# unrolling parameters into a single column vector
nn_initial_params = np.hstack((initial_theta1.ravel(order='F'), initial_theta2.ravel(order='F')))
#
#10
nn_backprop_Params = gradient(nn_initial_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0)
#11
checkGradient(nn_initial_params,nn_backprop_Params,input_layer_size, hidden_layer_size, num_labels,X,y,0)

#12
print('Gradient regularized, lambda = 1: ', gradient(nn_initial_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1))

#13
checkGradient(nn_initial_params,nn_backprop_Params,input_layer_size, hidden_layer_size, num_labels,X,y, 1)

#14
theta_opt = opt.fmin_cg(maxiter = 100, f = loss, x0 = nn_initial_params, fprime = gradient, args = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda))
theta1_opt = np.reshape(theta_opt[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), 'F')
theta2_opt = np.reshape(theta_opt[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1), 'F')

#15
y_pred = predict(theta1_opt, theta2_opt, X)
print("Accuracy: ", accuracy(y_pred, y))

#16


#17

lambdas = np.linspace(0,5, 20)
preds = []
for l in lambdas:
    theta_opt = opt.fmin_cg(maxiter=100, f=loss, x0=nn_initial_params, fprime=gradient,
                            args=(input_layer_size, hidden_layer_size, num_labels, X, y, l))
    theta1_opt = np.reshape(theta_opt[:hidden_layer_size * (input_layer_size + 1)],
                            (hidden_layer_size, input_layer_size + 1), 'F')
    theta2_opt = np.reshape(theta_opt[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1),
                            'F')
    y_pred = predict(theta1_opt, theta2_opt, X)
    preds.append(accuracy(y_pred, y))

plt.plot(lambdas, preds)
plt.xlabel("lambda", fontsize=14)
plt.ylabel("accuracy", fontsize=14)
plt.show()


