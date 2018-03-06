from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_Y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_Y)


# print(model.predict(data_X))
# y = 0.1x + 0.3 (0.1)
# print(model.coef_)
# 0.3
# print(model.intercept_)

# print(model.get_params)

# R^2 coefficient of determination
print(model.score(data_X, data_Y))
