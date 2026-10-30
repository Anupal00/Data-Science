import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance_data.csv")

X_train, X_test, y_train, y_test = train_test_split(df[["age", "affordability"]], df["bought_insurance"], test_size=0.2, random_state=25)

X_train["age"] = X_train["age"] / 100
X_test["age"] = X_test["age"] / 100

class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.b = 0

    def log_loss(self, y_true, y_pred):
        ep = 1e-15
        y_pred = np.clip(y_pred, ep, 1 - ep)
        return -np.mean((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, epochs, loss_threshold):
        self.w1, self.w2, self.b = self.gradient_descent(X["age"], X["affordability"], y, epochs, loss_threshold)
        return self.w1, self.w2, self.b

    def predict(self, X_test):
        w = self.w1 * X_test["age"] + self.w2 * X_test["affordability"] + self.b
        y_pred = self.sigmoid(w)
        return y_pred

    def gradient_descent(self, age, aff, y_true, epochs, loss_threshold):
        w1 = w2 = 1
        b = 0
        learning_rate = 0.1
        n = len(age)

        for epoch in range(epochs):
            w = w1 * age + w2 * aff + b
            y_pred = self.sigmoid(w)
            loss = self.log_loss(y_true, y_pred)

            w1_d = (1 / n) * np.dot(np.transpose(age), (y_pred - y_true))
            w2_d = (1 / n) * np.dot(np.transpose(aff), (y_pred - y_true))
            b_d = np.mean(y_pred - y_true)

            w1 = w1 - learning_rate * w1_d
            w2 = w2 - learning_rate * w2_d
            b = b - learning_rate * b_d

            if epoch % 50 == 0:
                print(f"Epoch : {epoch}, w1 : {w1}, w2 : {w2}, b : {b}, loss : {loss}")

            if loss < loss_threshold:
                print(f"Last Epoch : {epoch}, w1 : {w1}, w2 : {w2}, b : {b}, loss : {loss}")
                break
        return w1, w2, b



if __name__ == "__main__":
    custom_nn = myNN()
    w1, w2, b = custom_nn.fit(X_train, y_train, 100000, 0.4631)
    print(f"w1 : {w1}, w2 : {w2}, b : {b}")
    y_pred = custom_nn.predict(X_test)
    print(f"y_pred : {y_pred}")