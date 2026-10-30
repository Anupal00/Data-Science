import numpy as np
import math

y_true = np.array([1.0, 2.0, 3, 4])
y_predicted = np.array([1.1, 2.1, 3.1, 4.1])

def mae(y_true, y_predicted):
  total_error = 0
  for yt, yp in zip(y_true, y_predicted):
    total_error += abs(yt - yp)
  print(f"Total Error for MAE : {total_error}")
  mae = total_error / len(y_true)
  return mae

def mae_numpy(y_true, y_predicted):
  total_error = np.sum(np.abs(y_true - y_predicted))
  print(f"Total Error for MAE numpy : {total_error}")
  mae = total_error / len(y_true)
  return mae

def mse(y_true, y_predicted):
  total_error = 0
  for yt, yp in zip(y_true, y_predicted):
    total_error += abs(yt - yp) ** 2
  print(f"Total error for MSE : {total_error}")
  mse = total_error / len(y_true)
  return mse

def mse_numpy(y_true, y_predicted):
  total_error = np.sum(np.square(y_true - y_predicted))
  print(f"Total Error for MSE numpy : {total_error}")
  mse = total_error / len(y_true)
  return mse

def log_loss(y_true, y_predicted):
  ep = 1e-15
  y_predicted = np.array([max(i, ep) for i in y_predicted])
  y_predicted = np.array([min(1 - ep, i) for i in y_predicted])
  return -np.mean((y_true * np.log(y_predicted)) + ((1 - y_true) * np.log(1 - y_predicted)))

if __name__ == "__main__":
  x = mae(y_true, y_predicted)
  print(f"MAE : {x}")

  x_n = mae_numpy(y_true, y_predicted)
  print(f"MAE numpy : {x_n}")

  y = mse(y_true, y_predicted)
  print(f"MSE : {y}")

  y_n = mse_numpy(y_true, y_predicted)
  print(f"MSE numpy : {y_n}")

  z = log_loss(y_true, y_predicted)
  print(f"Log loss : {z}")