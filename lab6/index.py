import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def get_random_centoids(k, low, high):
    return np.random.randint(low, high, size=(k, 2))


def plotKmeans(X, centroids, idx, K, num_iters):
    """
    plots the data points with colors assigned to each centroid
    """
    m, n = X.shape[0], X.shape[1]
    for i in range(num_iters):
        # Visualisation of data
        color = "rgb"
        for k in range(1, K + 1):
            grp = (idx == k).reshape(m, 1)
            plt.scatter(X[grp[:, 0], 0], X[grp[:, 0], 1], c=color[k - 1], s=15)
        # visualize the new centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", c="black", linewidth=3)
        title = "Iteration Number " + str(i)
        plt.title(title)

        [centroids, idx] = k_means(X, idx, K, 1)
        # Compute the centroids mean
        # centroids = computeCentroids(X, idx, K)
        #
        # # assign each training example to the nearest centroid
        # idx = findClosestCentroids(X, centroids)
        plt.show()




def k_means(X, idx, K, num_iters):
    idx = findClosestCentroids(X, idx)
    for i in range(num_iters):
        # Compute the centroids mean
        centroids = computeCentroids(X, idx, K)

        # assign each training example to the nearest centroid
        idx = findClosestCentroids(X, centroids)

    return centroids, idx

def computeCentroids(X, idx, K):
    """
    returns the new centroids by computing the means of the data points assigned to each centroid.
    """
    m, n = X.shape[0], X.shape[1]
    centroids = np.zeros((K, n))
    count = np.zeros((K, 1))

    for i in range(m):
        index = int((idx[i] - 1)[0])
        centroids[index, :] += X[i, :]
        count[index] += 1

    return centroids / count

def findClosestCentroids(X, centroids):
    """
    Returns the closest centroids in idx for a dataset X where each row is a single example.
    """
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0], 1))
    temp = np.zeros((centroids.shape[0], 1))

    for i in range(X.shape[0]):
        for j in range(K):
            dist = X[i, :] - centroids[j, :]
            length = np.sum(dist ** 2)
            temp[j] = length
        idx[i] = np.argmin(temp) + 1
    return idx



#1
data = sio.loadmat('ex6data1.mat')
X = data.get('X')
plt.scatter(X [:,0], X [:,1])
plt.show()

K = 3
#2
init_centroids = get_random_centoids(K ,1, 6)
#3
idx = findClosestCentroids(X, init_centroids)
#4
centroids = computeCentroids(X, idx, K)
print("Centroids computed after initial finding of closest centroids:\n", centroids)
print(idx[:3])
#5
k_means(X,idx,K,10)

#6
m,n = X.shape[0],X.shape[1]
plotKmeans(X, init_centroids, idx, K,10)


#7
data = sio.loadmat('bird_small.mat')
X = data.get('A')


def kMeansInitCentroids(X, K):
    """
    This function initializes K centroids that are to beused in K-Means on the dataset X
    """
    m, n = X.shape[0], X.shape[1]
    centroids = np.zeros((K, n))

    for i in range(K):
        centroids[i] = X[np.random.randint(0, m + 1), :]

    return centroids
X2 = (X/255).reshape(128*128,3)
K2 = 16
num_iters = 10
#8
# preprocess and reshape the image
X2 = (X / 255).reshape(128 * 128, 3)
centroids2, idx2 = k_means(X2, kMeansInitCentroids(X2, K2), K2, num_iters)
m2,n2 = X.shape[0],X.shape[1]
X2_recovered = X2.copy()

#9
for i in range(1,K2+1):
    X2_recovered[(idx2==i).ravel(),:] = centroids2[i-1]
# Reshape the recovered image into proper dimensions
X2_recovered = X2_recovered.reshape(128,128,3)
# Display the image
import matplotlib.image as mpimg
fig, ax = plt.subplots(1,2)
ax[0].imshow(X2.reshape(128,128,3))
ax[1].imshow(X2_recovered)
plt.show()

