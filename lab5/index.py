import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np

def gauss_kernel(sigma):
    def gauss_kernel(x1, x2):
        sigma_squared = np.power(sigma, 2)
        matrix = np.power(x1-x2, 2)

        return np.exp(-np.sum(matrix)/(2*sigma_squared))

    return gauss_kernel

def dataset3Params(X, y, Xval, yval,vals):
    """
    Returns your choice of C and sigma. You should complete this function to return the optimal C and
    sigma based on a cross-validation set.
    """
    acc = 0
    best_c=0
    best_gamma=0
    for i in vals:
        C= i
        for j in vals:
            gamma = 1/j
            classifier = SVC(C=C,gamma=gamma)
            classifier.fit(X,y)
            prediction = classifier.predict(Xval)
            score = classifier.score(Xval,yval)
            if score>acc:
                acc =score
                best_c =C
                best_gamma=gamma
    return best_c, best_gamma
#1
data = sio.loadmat('ex5data1.mat')
X = data.get('X')
y = data.get('y')

#2

m, n = X.shape[0], X.shape[1]
pos, neg = (y == 1).reshape(m, 1).flatten(), (y == 0).reshape(m, 1).flatten()
plt.scatter(X[pos, 0], X[pos, 1])
plt.scatter(X[neg, 0], X[neg, 1])
plt.show()


#3
classifier = SVC(kernel="linear")
classifier.fit(X, y[:, 0])

#4
# plotting the decision boundary
X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
pos, neg = (y == 1).reshape(m, 1).flatten(), (y == 0).reshape(m, 1).flatten()
plt.scatter(X[pos, 0], X[pos, 1])
plt.scatter(X[neg, 0], X[neg, 1])
plt.xlim(0,4.5)
plt.ylim(1.5,5)
plt.show()


classifier = SVC(kernel="linear", C=100)
classifier.fit(X, y[:, 0])
X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
pos, neg = (y == 1).reshape(m, 1).flatten(), (y == 0).reshape(m, 1).flatten()
plt.scatter(X[pos, 0], X[pos, 1])
plt.scatter(X[neg, 0], X[neg, 1])
plt.xlim(0,4.5)
plt.ylim(1.5,5)
plt.show()



#6

data = sio.loadmat('ex5data2.mat')
X = data.get('X')
y = data.get('y')


#7-8-9
m2,n2 = X.shape[0],X.shape[1]
pos2,neg2= (y==1).reshape(m2,1), (y==0).reshape(m2,1)
classifier = SVC(kernel="rbf",gamma=30)
classifier.fit(X,y.ravel())

plt.figure(figsize=(8,6))
plt.scatter(X[pos2[:,0],0],X[pos2[:,0],1],c="r",marker="+")
plt.scatter(X[neg2[:,0],0],X[neg2[:,0],1],c="y",marker="o")
# plotting the decision boundary
X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,1)
plt.ylim(0.4,1)
plt.show()



#10

data = sio.loadmat('ex5data3.mat')
X = data["X"]
y = data["y"]
Xval = data["Xval"]
yval = data["yval"]
m3,n3 = X.shape[0],X.shape[1]
pos,neg= (y==1).reshape(m3,1), (y==0).reshape(m3,1)


#11

vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
C, gamma = dataset3Params(X, y.ravel(), Xval, yval.ravel(),vals)
classifier4 = SVC(C=C,gamma=gamma)
classifier4.fit(X,y.ravel())


plt.figure(figsize=(8,6))
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="y",marker="o",s=50)
# plotting the decision boundary
X_7,X_8 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_7,X_8,classifier4.predict(np.array([X_7.ravel(),X_8.ravel()]).T).reshape(X_7.shape),1,colors="b")
plt.xlim(-0.6,0.3)
plt.ylim(-0.7,0.5)
plt.show()