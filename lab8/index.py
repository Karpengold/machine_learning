import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def multivariateGaussian(X, mu, sigma2):
    k = len(mu)

    sigma2 = np.diag(sigma2)
    X = X - mu.T
    p = 1 / ((2 * np.pi) ** (k / 2) * (np.linalg.det(sigma2) ** 0.5)) * np.exp(
        -0.5 * np.sum(X @ np.linalg.pinv(sigma2) * X, axis=1))
    return p


def estimateGaussian(X):
    m = X.shape[0]
    sum_ = np.sum(X, axis=0)
    mu = 1 / m * sum_
    var = 1 / m * np.sum((X - mu) ** 2, axis=0)
    return mu, var


def selectThreshold(yval, pval):
    """
    Find the best threshold (epsilon) to use for selecting outliers
    """
    best_epi = 0
    best_F1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    epi_range = np.arange(pval.min(), pval.max(), stepsize)
    for epi in epi_range:
        predictions = (pval < epi)[:, np.newaxis]
        tp = np.sum(predictions[yval == 1] == 1)
        fp = np.sum(predictions[yval == 0] == 1)
        fn = np.sum(predictions[yval == 1] == 0)

        # compute precision, recall and F1
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        F1 = (2 * prec * rec) / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epi = epi

    return best_epi, best_F1


#1
data = sio.loadmat('ex8data1.mat')
X = data.get('X')
Xval = data.get('Xval')
yval = data.get('yval')

#2
plt.scatter(X[:,0], X[:, 1])
plt.show()

#3
mu, sigma2 = estimateGaussian(X)

#4
p = multivariateGaussian(X, mu, sigma2)


#5
# plt.figure(figsize=(8,6))
# plt.scatter(X[:,0],X[:,1],marker="x")
# X1,X2 = np.meshgrid(np.linspace(0,35,num=70),np.linspace(0,35,num=70))
# p2 = multivariateGaussian(np.hstack((X1.flatten()[:,np.newaxis],X2.flatten()[:,np.newaxis])), mu, sigma2)
# contour_level = 10**np.array([np.arange(-20,0,3,dtype=np.float)]).T
# plt.contour(X1,X2,p2[:,np.newaxis].reshape(X1.shape),contour_level)
# plt.xlim(0,35)
# plt.ylim(0,35)
# plt.xlabel("Latency (ms)")
# plt.ylabel("Throughput (mb/s)")
# plt.show()



#6

pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)
print("Best epsilon found using cross-validation:", epsilon)
print("Best F1 on Cross Validation Set:", F1)

#7
# Circling of anomalies
outliers = np.nonzero(p<epsilon)[0]
plt.scatter(X[outliers,0],X[outliers,1],marker ="o",facecolor="none",edgecolor="r",s=100)
plt.scatter(X[:,0], X[:, 1])
plt.show()


#8
data = sio.loadmat('ex8data2.mat')
X = data.get('X')
Xval = data.get('Xval')
yval = data.get('yval')


#9
# compute the mean and variance
mu, sigma2 = estimateGaussian(X)
print(mu, sigma2)
#10
# Training set
p = multivariateGaussian(X, mu, sigma2)
# cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2)

# Find the best threshold
epsilon2, F1_2 = selectThreshold(yval, pval)
print("Best epsilon found using cross-validation:",epsilon2)
print("Best F1 on Cross Validation Set:",F1_2)
print("# Outliers found:",np.sum(p<epsilon2))

