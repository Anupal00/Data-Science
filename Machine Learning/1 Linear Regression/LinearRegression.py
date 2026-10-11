import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns

df = pd.read_csv('homeprices.csv')
print(df.head())

plt.xlabel("area")
plt.ylabel("price")
plt.scatter(df['area'], df['price'], color='red', marker='o')