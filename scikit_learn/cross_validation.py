import numpy as np
from sklearn import datasets
from sklearn.cross_validation import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target


k_range = range(1, 31)
k_scores = []

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  scores = -cross_val_score(knn, iris_X, iris_Y, cv=10, scoring='mean_squared_error') #for regression
  # scores = cross_val_score(knn, iris_X, iris_Y, cv=10, scoring='accuracy') # for classification
  k_scores.append(scores.mean())


plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()