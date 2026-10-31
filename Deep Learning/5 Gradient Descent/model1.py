import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./price.csv")

X_train, X_test, y_train, y_test = train_test_split(df[["area", "bedrooms"]], df[["price"]], test_size=0.2)
scaler = MinMaxScaler()

scaled_x = scaler.fit_transform(X_train)
scaled_y = scaler.fit_transform(y_train)

print(f"scaled x shape : {scaled_x.shape}")
print(f"scaled y shape : {scaled_y.shape}")


def batch_gradient_descent_mse(X, y_true, epochs, learning_rate=0.1):
    w = np.ones(shape=X.shape[1])
    total_elements = len(X)
    b = 0
    loss_mse = 0
    for e in range(epochs):
        y_pred = np.dot(X, w) + b
        error = y_pred - y_true

        wd = (2 / total_elements) * np.dot(np.transpose(X), error)
        bd = (2 / total_elements) * np.sum(error)

        loss_mse = np.mean(np.square(error))

        w = w - (learning_rate * wd)
        b = b - (learning_rate * bd)

    return w, b, loss_mse


def batch_gradient_descent_mae(X, y_true, epochs, learning_rate=0.1):
    w = np.ones(shape=X.shape[1])
    total_elements = len(X)
    b = 0;
    loss_mae = 0;

    for e in range(epochs):
        y_pred = np.dot(X, w) + b
        error = y_pred - y_true

        wd = (1 / total_elements) * np.dot(np.transpose(X), np.sign(error))
        bd = (1 / total_elements) * np.sum(np.sign(error))

        loss_mae = np.mean(np.abs(error))

        w = w - (learning_rate * wd)
        b = b - (learning_rate * bd)

    return w, b, loss_mae


def SGD(X, y_true, epochs, learning_rate=0.1):
    w = np.ones(shape=X.shape[1])
    b = 0
    loss = 0
    total_element = len(X)

    for e in range(epochs):
        pos = np.random.randint(0, total_element)
        X_s = X[pos]
        y_s = y_true[pos]
        y_pred = np.dot(X_s, w) + b

        error = y_pred - y_s

        wd = 2 * X_s * error
        bd = 2 * error

        loss = np.square(error)

        w = w - (learning_rate * wd)
        b = b - (learning_rate * bd)

    return w, b, loss


if __name__ == "__main__":
    w, b, loss_mse = batch_gradient_descent_mse(scaled_x, scaled_y.flatten(), 5)
    print(f"w : {w} , b : {b}, loss_mse : {loss_mse}")

    w1, b1, loss_mae = batch_gradient_descent_mae(scaled_x, scaled_y.flatten(), 5)
    print(f"w1 : {w1} , b1 : {b1}, loss_mse : {loss_mae}")

    w2, b2, loss = SGD(scaled_x, scaled_y.flatten(), 5)
    print(f"SGD w2 : {w1} , b2 : {b1}, loss : {loss}")