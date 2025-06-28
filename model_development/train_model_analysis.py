import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats

# Load and preprocess
# Get path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full path to the CSV
csv_path = os.path.join(script_dir, 'demanddataupdate.csv')
# Read the CSV
df = pd.read_csv(csv_path)

# Filter & feature engineering
df = df[df['FORECAST_ACTUAL_INDICATOR'] == 'A']
df = df[['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND']]
df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
df['time'] = pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 0.5, unit='h')
df['datetime'] = df['SETTLEMENT_DATE'] + df['time']
df.sort_values('datetime', inplace=True)
df['month'] = df['datetime'].dt.month
df['dayofweek'] = df['datetime'].dt.weekday

# Split into train and test
split_index = int(len(df) * 0.8)
df_train = df.iloc[:split_index].copy()
df_test = df.iloc[split_index:].copy()

# Add lags to training set
df_train['lag_1'] = df_train['ND'].shift(1)
df_train['lag_48'] = df_train['ND'].shift(48)
df_train.dropna(inplace=True)

features = ['SETTLEMENT_PERIOD', 'month', 'dayofweek', 'lag_1', 'lag_48']
X_train = df_train[features]
y_train = df_train['ND']

# Train Ridge model
ridge_model = RidgeCV()
ridge_model.fit(X_train, y_train)

# Rolling forecast
ridge_preds, naive_preds, actuals = [], [], []
test_dates = df_test['SETTLEMENT_DATE'].unique()
history = df_train.copy()

for test_day in test_dates:
    test_block = df_test[df_test['SETTLEMENT_DATE'] == test_day].copy()
    prev_day = test_day - pd.Timedelta(days=1)
    prev_day_data = history[history['SETTLEMENT_DATE'] == prev_day]

    if prev_day_data.empty or len(prev_day_data) < 48:
        continue

    prev_day_mean = prev_day_data['ND'].mean()
    day_preds_ridge = []

    for i, row in test_block.iterrows():
        lag_1 = history['ND'].iloc[-1] if len(day_preds_ridge) == 0 else day_preds_ridge[-1]
        lag_48_index = -48 + len(day_preds_ridge)
        lag_48 = history['ND'].iloc[lag_48_index] if len(history) >= 48 else lag_1

        X_row = pd.DataFrame([[
            row['SETTLEMENT_PERIOD'],
            row['month'],
            row['dayofweek'],
            lag_1,
            lag_48
        ]], columns=features)

        pred = ridge_model.predict(X_row)[0]
        day_preds_ridge.append(pred)

    ridge_preds.extend(day_preds_ridge)
    naive_preds.extend([prev_day_mean] * len(test_block))
    actuals.extend(test_block['ND'].values)

    test_block['ND'] = test_block['ND'].values
    history = pd.concat([history, test_block], ignore_index=True)

# Convert to arrays
ridge_preds = np.array(ridge_preds)
naive_preds = np.array(naive_preds)
actuals = np.array(actuals)

# Evaluation
ridge_rmse = np.sqrt(mean_squared_error(actuals, ridge_preds))
naive_rmse = np.sqrt(mean_squared_error(actuals, naive_preds))
ridge_mae = mean_absolute_error(actuals, ridge_preds)
naive_mae = mean_absolute_error(actuals, naive_preds)

print("\nEvaluation Metrics:")
print(f"Ridge RMSE: {ridge_rmse:.4f} | MAE: {ridge_mae:.4f}")
print(f"Naive RMSE: {naive_rmse:.4f} | MAE: {naive_mae:.4f}")

# Plot results
plt.figure(figsize=(16, 6))
plt.plot(actuals, label='Actual', color='black', linewidth=2)
plt.plot(ridge_preds, label='Ridge Forecast', linestyle='-', markersize=2)
plt.plot(naive_preds, label='Naive Forecast (yesterday avg)', linestyle='--')
plt.title("Forecast Comparison: Ridge vs Naive vs Actual (All Test Days)")
plt.xlabel("Time Step (Half-Hours)")
plt.ylabel("Demand (ND)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Diebold-Mariano Test
def diebold_mariano_test(e1, e2, power=1):
    d = np.abs(e1)**power - np.abs(e2)**power
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var == 0:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * stats.norm.cdf(-abs(dm_stat))
    return dm_stat, p_value

ridge_errors = actuals - ridge_preds
naive_errors = actuals - naive_preds
dm_stat, p_value = diebold_mariano_test(ridge_errors, naive_errors)

print("\nDiebold-Mariano Test (Absolute Error):")
print(f"Sample size: {len(actuals)}")
print(f"DM statistic: {dm_stat:.4f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("→ Statistically significant difference in forecast accuracy.")
else:
    print("→ No statistically significant difference in forecast accuracy.")