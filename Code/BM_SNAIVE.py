import warnings
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


S = 12              # seasonal period (months)

df = pd.read_csv('Data/main_dataset.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date').asfreq('MS')

# Monthly series (target)
series = df['volume'].astype(float)
assert series.isna().sum() == 0, "Missing months in volume"

# Split dates (same as ARIMAX, SARIMAX and the deep learning models)
train_end = "2023-04-01"
val_end = "2024-04-01"

train = series[series.index < train_end]
val = series[(series.index >= train_end) & (series.index < val_end)]
test = series[series.index >= val_end]

# Print ranges
print("Train:", train.index.min(), "->", train.index.max())
print("Val:  ", val.index.min(), "->", val.index.max())
print("Test: ", test.index.min(), "->", test.index.max())

# Number of observations
print("\nNumber of observations:")
print("Train:", len(train))
print("Val:  ", len(val))
print("Test: ", len(test))
print("Seasonal period:", S)

seasonal_naive = series.shift(S)

val_pred = seasonal_naive.loc[val.index]
forecast = seasonal_naive.loc[test.index]
assert forecast.notna().all(), "Not enough history for the seasonal naive forecast"


# -------------------------------------
# 2. Validation metrics (for reference only; there is nothing to tune)
# -------------------------------------
val_rmse = np.sqrt(mean_squared_error(val, val_pred))
print(f"\nValidation RMSE = {val_rmse:.2f}")


# -------------------------------------
# 3. Future dates (same as the ARIMAX code)
# -------------------------------------
n_steps = len(test)
split_date_dt = pd.to_datetime('2024-04-01')

future_dates = pd.date_range(start=split_date_dt,
                             periods=n_steps,
                             freq='MS')
assert (future_dates == test.index).all()

forecast.index = future_dates
forecast.name = 'predicted_arrivals'

print(forecast)
forecast.to_csv("Results/test_predicted_snaive.csv")   # same file format as ARIMAX / SARIMAX


# -------------------------------------
# 4. Metrics
# -------------------------------------
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100
print(f"Test  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

#pd.DataFrame([{"model": "Seasonal naive", "season": S, "MAE": mae, "RMSE": rmse, "MAPE": mape}]).to_csv("Results/snaive_test_metrics.csv", index=False)

with open("Results/best_hparams_snaive.json", "w") as f:
    json.dump({"season": S, "val_RMSE": float(val_rmse)}, f)