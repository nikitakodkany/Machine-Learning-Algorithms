import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from numpy.linalg import norm


from numpy.linalg import norm
#load dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target
#y = y.reshape(150,1)

#plot the dataset
plt.scatter(x[:,0], x[:,1], c = 'black', label = 'unclustered data')
plt.xlabel('Training Examples')
plt.ylabel('Features')
plt.title('Plot of data points')
plt.show()


#split the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x,y, train_size=0.8, test_size=0.2, random_state=2)

#feature scaling and normalization
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

#number of clusters
k = 3

#generate randome centers
mean = np.mean(xtrain, axis = 0)
std = np.std(xtrain, axis = 0)
centroids = np.random.randn(k,xtrain.shape[1])*std + mean

centroids_old = np.zeros(centroids.shape)
centroids_new = deepcopy(centroids)

centroids = np.zeros(xtrain.shape[0])
distances = np.zeros((xtrain.shape[0],k))

error = norm(centroids_new - centroids_old)

while error !=0:
    #measure distance for every center
    for i in range(k):
        distances[:,i] = norm(xtrain - centroids_new[i], axis=1)
    #assign all training data to closest center
    clusters = np.argmin(distances, axis=1)

    centroids_old = deepcopy(centroids_new)
    #calculate mean for every cluster and update the center
    for i in range(k):
        centroids_new[i] = np.mean(xtrain[clusters == i], axis=0)
    error = norm(centroids_new - centroids_old)
centroids_new


#plot the dataset
colors = ['orange', 'blue', 'green']
for i in range(xtrain.shape[0]):
    plt.scatter(xtrain[i,0], xtrain[i,1], s=7, color = colors[int(ytrain[i])])
plt.scatter(centroids_new[:,0], centroids_new[:,1], marker='*', c='g', s=150)

plt.show()
