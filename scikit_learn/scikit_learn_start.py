import numpy as np
#sklearn's databases
from sklearn import datasets
from sklearn.cross_validation import train_test_split

# select a nearby point to generate the model
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
# property
iris_X = iris.data
# classification
iris_Y = iris.target

# print(iris_X[:2, :])
# print(iris_Y)

# separating test data and train data, test_size means that the test data accounts for 30% of all data
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=0.3)

print(Y_test)
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)

# print(knn.predict(X_test))
# print(Y_test)