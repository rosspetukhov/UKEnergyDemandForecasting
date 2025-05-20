import pandas as pd
import os
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Get path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full path to the CSV
csv_path = os.path.join(script_dir, 'demanddataupdate.csv')

# Read the CSV
df = pd.read_csv(csv_path)

# Filter relevant rows and columns
df = df.loc[df['FORECAST_ACTUAL_INDICATOR'] == 'A']
df = df[['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND']]

# Convert dates and build datetime
df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
df['time'] = pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 0.5, unit='h')  # SP=1 means 00:00
df['datetime'] = df['SETTLEMENT_DATE'] + df['time']

# Add features
df['month'] = df['datetime'].dt.month
df['dayofweek'] = df['datetime'].dt.weekday
df['lag_1'] = df['ND'].shift(1)
df['lag_48'] = df['ND'].shift(48)

# Remove rows with missing data
df.dropna(inplace=True)

# Features and labels
features = ['SETTLEMENT_PERIOD', 'month', 'dayofweek', 'lag_1', 'lag_48']
X = df[features]
y = df['ND']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

# Train model
model = RidgeCV()
model.fit(X_train, y_train)

# Save the model to a file
with open('model_development/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Forecast next 48 half-hour periods 
last_datetime = df['datetime'].iloc[-1]
future_preds = []
recent_values = df.copy()

for i in range(1, 49):  
    future_time = last_datetime + pd.Timedelta(minutes=30 * i)
    settlement_period = ((future_time.hour * 60 + future_time.minute) // 30) + 1

    month = future_time.month
    dayofweek = future_time.weekday()
    lag_1 = future_preds[-1] if len(future_preds) > 0 else recent_values.iloc[-1]['ND']
    lag_48 = recent_values.iloc[-48 + i]['ND'] if len(recent_values) >= 48 else lag_1

    X_future = pd.DataFrame([[settlement_period, month, dayofweek, lag_1, lag_48]], columns=features)
    pred = model.predict(X_future)[0]
    future_preds.append(pred)

# Plot the forecast
plt.plot(range(1, 49), future_preds, marker='o')
plt.title('Next 24-Hour Forecast (48 half-hour periods)')
plt.xlabel('Half-Hour Period Ahead')
plt.ylabel('Predicted ND')
plt.grid(True)
plt.tight_layout()
plt.show()