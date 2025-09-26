
# Линейная регрессия
# import numpy as np
# from sklearn.linear_model import LinearRegression

# dist = np.array([[50], [100], [150], [200], [250]])
# costs = np.array([1.5, 3.0, 4.5, 6.0, 7.5])

# model = LinearRegression()
# model.fit(dist, costs)

# dist_new = np.array([[300], [350], [400]])
# costs_pred = model.predict(dist_new)

# print(costs_pred)

#Логистическая регрессия
# import numpy as np
# from sklearn.linear_model import LogisticRegression

# x = np.array([[22, 33], [25, 50], [28, 70], [35, 60], [40, 80]])
# y = np.array([0, 0, 1, 1, 1])

# model = LogisticRegression()
# model.fit(x, y)

# x_new = np.array([[30, 65], [23, 40]])
# y_prob = model.predict_proba(x_new)
# y_pred = model.predict(x_new)
# print(y_prob, y_pred)
















import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[5, 1], [10, 2], [15, 3], [20, 5], [25, 4]])
y = np.array([0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(x, y)

x_new = np.array([[12, 2], [18, 4], [6, 1]])
y_prob = model.predict_proba(x_new)
y_pred = model.predict(x_new)

print(y_prob)
print(y_pred)