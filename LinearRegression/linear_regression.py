import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("linear_regression_data.csv").values   # Convert DataFrame to NumPy array

# Custom train_test_split function
def train_test_split(data, test_size=0.2, shuffle=True):

    if shuffle:
        np.random.shuffle(data)

    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)

    split_index = int(len(data) * (1 - test_size))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(df, test_size=0.2, shuffle=True)


def cost_function(X, y, m, c):
    n = X.shape[0]
    y_pred = m * X + c
    cost = (1 / n) * np.sum((y - y_pred) ** 2)
    return cost

# Use the property of gradients that negaitive of gradient always points in the direction of steepest decrease
def gradient_descent(X_train, y_train, learning_rate, m, c):
    n = X_train.shape[0]
    derivative_wrt_m = 0
    derivative_wrt_c = 0
 
    for i in range(n):
        x = X_train[i]
        y = y_train[i]
        error = y - (m * x + c)
        derivative_wrt_m -= (2 / n) * error * x
        derivative_wrt_c -= (2 / n) * error

    m -= learning_rate * derivative_wrt_m
    c -= learning_rate * derivative_wrt_c

    return m, c


def plot(data, m, c):
    plt.scatter(data[:, 0], data[:, 1], label="Data", color="blue")
    
    x_range = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
    y_pred = m * x_range + c

    plt.plot(x_range, y_pred, color="red", label="Regression Line")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.show()

"""
    R^2 score = 1 - summation((y_true - y_pred)^2)/summation((y_true - y_mean)^2)

    Significance: It tells us how well our model explains the variance of data

    Interpretation:
    R^2 = 1.0 : Perfect score(Model explains variance perfectly)
    R^2 = 0.0 : Model is as good as using the mean of y as a prediction
    R^2 < 0.0 : Model is worse than using mean of y(Basically worse than guessing)

    ** Why not use accuracy instead?? **
    --> Data is continuous, hence using classical accuracy does not work in this case
"""
def r2_score(X_test, y_true, m, c):
    y_pred = m * X_test + c
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def main():
    "Try using different epochs and learning rates and compare their R^2 scores"
    epoch = 10000
    learning_rate = 0.0001
    m, c = 0, 0
    cost_history = []
    for _ in range(epoch):
        m, c = gradient_descent(X_train, y_train, learning_rate, m, c)
        cost = cost_function(X_train, y_train, m, c)
        cost_history.append(cost)


    print(f"Final Parameters: m = {m}    c = {c}")
    print("R² Score:", r2_score(X_test, y_test, m, c))

    plot(df, m, c)

    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Epochs")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Function Over Epochs")
    plt.show()

main()
