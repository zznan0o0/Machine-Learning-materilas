from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# loaded_data = datasets.load_boston()
# data_X = loaded_data.data
# data_Y = loaded_data.target

# model = LinearRegression()
# model.fit(data_X, data_Y)

# print(model.predict(data_X))
# print(data_Y)


X, Y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
plt.scatter(X, Y)
plt.show()