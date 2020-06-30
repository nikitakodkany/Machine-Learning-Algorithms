import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
x = iris.data
y = iris.target

#plot the dataset
colors = ['blue', 'orange', 'green']
for i in range(x.shape[0]):
    plt.scatter(x[i,0], x[i,1], s=7, color = colors[int(y[i])])
plt.show()


xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size = 0.8, test_size = 0.2)

sc = StandardScaler()
sc.fit(xtrain)
xtrain = sc.transform(xtrain)
xtest = sc.transform(xtest)

def euclidian_distance(vector1, vector2):
    for i in range (len(vector1-1)):
        #vector1 is first row of data and vector2 is second row of data
        return np.sqrt(np.sum((vector1-vector2)**2))

def kneighbours(xtrain, xtest_row, k):
    distance = []
    neighbours = []
    for i in range(0,xtrain.shape[0]):
        dist = euclidian_distance(xtrain[i], xtest_row)
        distance.append((i,dist))
    distance.sort(key=lambda x: x[1])
    for j in range(k):
        neighbours.append(distance[j][0])
    return neighbours

def predict_classification(neighbours, ytrain):
    votes = {}
    for i in range(len(neighbours)):
        if ytrain[neighbours[i]] in votes:
            votes[ytrain[neighbours[i]]] += 1
        else:
            votes[ytrain[neighbours[i]]] = 1
    sorted_votes = sorted(votes.items(), key=lambda x: x[1],reverse = True )
    return sorted_votes[0][0]

def kNN(xtrain, xtest, ytrain, ytest, k):
    output_classes = []
    for i in range(0, xtest.shape[0]):
        neighbours = kneighbours(xtrain, xtest[i], k)
        predictedClass = predict_classification(neighbours, ytrain)
        output_classes.append(predictedClass)
    return output_classes

def accuracy(yhat, ytest):
    count = 0
    for i in range(len(ytest)):
        if yhat[i] == ytest[i]:
            count += 1
    return (count)/len(yhat)*100

k = 6
predicted_classes = {}
predicted_classes[k] = kNN(xtrain, xtest, ytrain, ytest, k)

accuracy = accuracy(predicted_classes[k], ytest)
print("Accuracy of classification: ", accuracy)

from sklearn.metrics import classification_report
print("Classification report")
print(classification_report(ytest,predicted_classes[k]))
