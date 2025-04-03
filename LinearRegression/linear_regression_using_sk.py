import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("linear_regression_data.csv")
df = np.array(df)
X_train, X_test, y_train, y_test = train_test_split(df[:,0].reshape(-1,1), df[:, 1].reshape(-1,1), test_size=0.2, train_size=0.8)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
plt.scatter(df[:,0], df[:,1])
plt.plot(np.linspace(0, 20, 50).reshape(-1,1), model.predict(np.linspace(0, 20, 50).reshape(-1,1)), 'r')
plt.show()
