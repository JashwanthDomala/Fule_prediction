import pandas as pd

df = pd.read_csv("petrol_price.csv")

df.columns = df.columns.str.strip()

print(df.columns)
df = df[['date', 'Delhi']]
df.columns = ['date', 'price']
df['date'] = pd.to_datetime(df['date'], format='%Y_%b')
df = df.sort_values('date')
print(df.isnull().sum())
df = df.dropna()
print(df.head())
print(df.tail())
print(df.dtypes)

import matplotlib.pyplot as plt

# df.plot(x='date', y='price', title="Petrol Price Trend")
# plt.show()

df.to_csv("clean_petrol.csv", index=False)

df.set_index('date', inplace=True)

df = df.asfreq('MS')

df = df.ffill()



df = df.reset_index()

df.columns = ['ds', 'y']
print(df.head())
print(df.dtypes)
df.to_csv("prophet_ready.csv", index=False)



# model

from prophet import Prophet

model = Prophet()

model.fit(df)

future = model.make_future_dataframe(periods=60, freq='MS')

forecast = model.predict(future)

print(forecast[['ds', 'yhat']].tail())

import matplotlib.pyplot as plt

model.plot(forecast)
plt.show()


# training 

# ===== DAY 5: TRAIN TEST SPLIT =====

train_size = int(len(df) * 0.8)

train = df[:train_size]
test = df[train_size:]

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=len(test), freq='MS')

forecast = model.predict(future)

forecast_test = forecast[['ds', 'yhat']].tail(len(test))

comparison = test.merge(forecast_test, on='ds')

print(comparison.head())

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(comparison['y'], comparison['yhat'])
rmse = np.sqrt(mean_squared_error(comparison['y'], comparison['yhat']))

print("MAE:", mae)
print("RMSE:", rmse)

import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)