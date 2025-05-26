import pandas as pd
import os
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
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
with open('model_development/ridge_model_24may.pkl', 'wb') as f:
    pickle.dump(model, f)

