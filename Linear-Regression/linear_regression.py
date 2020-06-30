#import libraries
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#import dataset
boston = datasets.load_boston()
x = boston.data #506, 13
y = boston.target #506, 1

#split the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x,y, train_size=0.8, test_size=0.2, random_state=2)
ytrain = ytrain.reshape(404,1)
ytest = ytest.reshape(102,1)

#feature scaling
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

class linear_regression():
    """Docstring for Linear Regression"""

    def __init__(self, xtrain, ytrain):
        self.cache = {}
        np.random.seed(2)
        self.cache['x'] = xtrain
        self.cache['y'] = ytrain
        self.cache['m'] = xtrain.shape[0]
        self.cache['t'] = np.zeros((xtrain.shape[1],1))

    def cost(self, error, y=None):

        if y is None:
            m = self.cache['m']
            y = self.cache['y']

        cost = (1/(2*m) * np.sum(((error-y)**2))) / y.shape[0]

        return cost

    def gradient_descent(self,y = None):
        if y is None:
            x = self.cache['x']
            y = self.cache['y']
            t = self.cache['t']
            m = self.cache['m']

        lr = 0.01

        a0 = np.dot(x,t) #401,13 . 13,1
        error = a0 - y #401,1

        cost = self.cost(error)
        t -= (lr * (1/m) * np.dot(x.T,(error - y))) #13,401 . 401,1

        return cost, t

def main():

    epoch = 1000
    reg = linear_regression(xtrain, ytrain)
    costs = []
    thetas = []

    for i in range(epoch):
        cost, theta = reg.gradient_descent()
        costs.append(cost)
        thetas.append(theta)



    #Plot Graph
    plt.title('Cost Function')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()
