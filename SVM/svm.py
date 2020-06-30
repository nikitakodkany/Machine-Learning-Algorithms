#libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#load dataset
cancer = datasets.load_breast_cancer()
x = cancer.data #569,30
y = cancer.target
y = y.reshape(569,1)

#splitting the datasets
xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.8, test_size=0.2)
#xtrain (455,30)

#feature scaling
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

#initialization
C = 10000000 #regularization parameter
lr = 0.000001 #learning rate

def ccost(w,xtrain,ytrain):
    # w 30,1
    a = 1 - (np.multiply(ytrain,np.dot(xtrain,w))) #455,1
    b = np.maximum(0,a) #455,1
    hinge_loss = C * (np.sum(b))

    #calculate cost
    cost = ((1/2 * np.dot(w.T,w)) + hinge_loss).flatten().tolist().pop() #1,1
    return cost

def costgradient(w, xtrain, ytrain):
    #subgradient
    xtrain = xtrain.reshape(30,1)
    ytrain = ytrain.reshape(1,1)
    w = w.reshape(30,1)
    dw = np.zeros(w.shape)

    a = 1 - (np.multiply(ytrain,np.dot(xtrain.T,w))) #1,1
    for i, j in enumerate(a):
        j = np.maximum(0,a)
        if j.any() == 0:
            di = w
        else:
            di = w - (C * ytrain[i] * xtrain[i]) #30,1

        dw += di
    dw = dw/len(ytrain)
    return dw

def sgd(xtrain, ytrain):
    w = np.zeros(xtrain.shape[1])
    w = w.reshape(30,1)
    max_epoch = 100
    costlist = []

    for epoch in range(1, max_epoch):
        xtrain, ytrain = shuffle(xtrain, ytrain)
        for i, j in enumerate(xtrain):
            a = costgradient(w, xtrain[i], ytrain[i])
            w = w - (lr * a)

        cost = ccost(w, xtrain, ytrain)
        costlist.append(cost)
        print("Epoch: {}  Cost: {}".format(epoch,cost))

    return costlist, w

def svm_model():
    costlist = []
    print("Training Started...")
    costlist, w = sgd(xtrain, ytrain)
    print("Training Complete...")


    #accuracy
    yhat = np.array([])
    for i in range(xtest.shape[0]):
        yp = np.sign(np.dot(w.T, xtest[i]))
        yhat = np.append(yhat,yp)
    yhat = yhat.ravel()
    yhat = yhat.tolist()
    for i in range(len(yhat)):
        if yhat[i] == -1.0:
            yhat[i] = 0
        else:
            yhat[i] = 1
    print(classification_report(ytest,yhat))
     
    plt.plot(costlist)
    plt.show()



svm_model()
