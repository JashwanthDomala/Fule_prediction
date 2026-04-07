import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ================= LOAD =================

df = pd.read_csv("petrol_price.csv")

df.columns = df.columns.str.strip()

df = df[['date', 'Delhi']]
df.columns = ['date', 'price']

df['date'] = pd.to_datetime(df['date'], format='%Y_%b')

df = df.sort_values('date')

df = df.drop_duplicates(subset='date', keep='last')

df = df.dropna()

# ================= TIME SERIES =================

df.set_index('date', inplace=True)

df = df.asfreq('MS')

df['price'] = df['price'].interpolate()

df = df.reset_index()

# ================= FEATURE ENGINEERING =================

df['time_index'] = np.arange(len(df))

# Polynomial features (curve instead of straight line)
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(df[['time_index']])

y = df['price']

# ================= TRAIN =================

model = LinearRegression()
model.fit(X, y)

print("Model trained")

# ================= FUTURE =================

future_periods = 60

future_index = np.arange(len(df), len(df) + future_periods)

future_X = poly.transform(future_index.reshape(-1, 1))

future_pred = model.predict(future_X)

# ================= PLOT =================

plt.figure(figsize=(10,5))

plt.plot(df['date'], df['price'], label="Actual")
future_dates = pd.date_range(start=df['date'].iloc[-1], periods=future_periods+1, freq='MS')[1:]

plt.plot(future_dates, future_pred, label="Prediction", color='green')

plt.legend()
plt.title("Petrol Price Prediction (Improved)")
plt.show()

# ================= SAVE =================

import pickle

with open("model.pkl", "wb") as f:
    pickle.dump((model, poly), f)

print("✅ Model saved")