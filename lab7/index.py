import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def cov(X):
    return X.T.dot(X) / X.shape[0]

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def pca(X):
    U, S, V = np.linalg.svd(cov(X))
    return U, S, V


def projectData(X, U, K):
    U_reduced = U[:, :K]
    Z= np.dot(X, U_reduced)
    return Z


def recoverData(Z, U, K):
    U_reduced = U[:, :K]
    X_rec =np.dot( Z , U_reduced.T)
    return X_rec
#1
data = sio.loadmat('ex7data1.mat')
X = data.get('X')

#2
plt.scatter(X [:,0], X [:,1])
plt.show()


X_norm,mu,std = featureNormalize(X)


#3
U,S = pca(X_norm)[:2]

#4-5
plt.scatter(X[:,0],X[:,1])
plt.plot([mu[0],(mu+1.5*S[0]*U[:,0].T)[0]],[mu[1],(mu+1.5*S[0]*U[:,0].T)[1]],color="black",linewidth=3)
plt.plot([mu[0],(mu+1.5*S[1]*U[:,1].T)[0]],[mu[1],(mu+1.5*S[1]*U[:,1].T)[1]],color="black",linewidth=3)
plt.xlim(-1,7)
plt.ylim(2,8)
plt.show()

#6
# Project the data onto K=1 dimension
K=1
Z = projectData(X_norm, U, K)
print("Projection of the first example:",Z[0][0])


#7
X_rec  = recoverData(Z, U, K)


#8

plt.scatter(X_norm[:,0],X_norm[:,1],marker="o",label="Original",facecolors="none",edgecolors="b",s=15)
plt.plot(X_rec[:,0],X_rec[:,1], label="Projection")
plt.title("The Normalized and Projected Data after PCA")
plt.legend()
plt.show()

#9

data = sio.loadmat('ex7faces.mat')
X = data.get('X')
fig, ax = plt.subplots(nrows=10,ncols=10,figsize=(8,8))
indexes = np.random.randint(0, 4999, 100)
axis = 0
for i in indexes:
    for j in range(10):
        ax[int(axis/10),j].imshow(X[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax[int(axis/10),j].axis("off")
    axis +=1
plt.show()


X_norm = featureNormalize(X)[0]

U =pca(X_norm)[0]
#Visualize the top 36 eigenvectors found
U_reduced = U[:,:36].T
fig2, ax2 = plt.subplots(6,6,figsize=(8,8))
for i in range(0,36,6):
    for j in range(6):
        ax2[int(i/6),j].imshow(U_reduced[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax2[int(i/6),j].axis("off")
plt.show()


U_reduced = U[:,:100].T
fig2, ax2 = plt.subplots(10,10,figsize=(8,8))
for i in range(0,100,10):
    for j in range(10):
        ax2[int(i/10),j].imshow(U_reduced[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax2[int(i/10),j].axis("off")
plt.show()