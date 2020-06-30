import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#load dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target
y = y.reshape(150,1)

#plot the dataset
colors = ['orange', 'blue', 'green']
for i in range(x.shape[0]):
    plt.scatter(x[i,0], x[i,1],  s=7, c= colors[int(y[i])])
plt.title('Plot of data points')
plt.show()

#split the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x,y, train_size=0.8, test_size=0.2, random_state=2)
x = xtrain #120,4
y = ytrain #120,1

#standardization
sc = StandardScaler()
x = sc.fit_transform(x)
xtest = sc.fit_transform(xtest)


covariance_matrix = np.cov(x.T) #4,4
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix) #4 | 4,4


#plot - variance explained ratios of Eigen Values
tot = sum(eigen_values)
var_exp = [(i / tot) for i in sorted(eigen_values, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,5), var_exp, alpha=0.5,
        align='center', label='individual explained variance')
plt.step(range(1,5), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()


# Make a list of (eigen value, eigen vector) tuples
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
# Sort the (eigen value, eigen vector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

#feature_vector
feature_vector= np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis])) #4,2
#transformed_data
x_pca = np.dot(x,feature_vector) #120,2


#plot PCA
for i in range(x_pca.shape[0]):
    plt.scatter(x_pca[i,0],x_pca[i,1], s=6,c=colors[int(ytrain[i])-1])
plt.show()



#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_pca, ytrain.ravel())

xtest_pca=np.dot(xtest,feature_vector)
# Predicting the Test set results
y_pred = classifier.predict(xtest_pca)


#Classification Report
from sklearn.metrics import classification_report
print("Classification report")
print(classification_report(ytest,y_pred))
