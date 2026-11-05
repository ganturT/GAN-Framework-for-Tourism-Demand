import warnings
import itertools
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


FEATS = ["search", "sin12", "cos12", "covid"]
NOWCAST = True      # True: this month's search (lag 0); False: last month's search


df = pd.read_csv('Data/main_dataset.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date').asfreq('MS')


series = df['volume'].astype(float)
exog = df[FEATS].astype(float).copy()

if not NOWCAST:
    exog['search'] = exog['search'].shift(1)          # last month's search
    ok = exog.notna().all(axis=1)
    series, exog = series[ok], exog[ok]

assert series.isna().sum() == 0, "Missing months in volume"
assert exog.isna().sum().sum() == 0, "Missing values in exogenous features"

# Split dates
train_end = "2023-04-01"
val_end = "2024-04-01"

train = series[series.index < train_end]
val = series[(series.index >= train_end) & (series.index < val_end)]
test = series[series.index >= val_end]

# Standardise the features with TRAIN statistics only
mu = exog.loc[train.index].mean()
sd = exog.loc[train.index].std().replace(0, 1.0)
exog = (exog - mu) / sd

X_train = exog.loc[train.index]
X_val = exog.loc[val.index]
X_test = exog.loc[test.index]

# Print ranges
print("Train:", train.index.min(), "->", train.index.max())
print("Val:  ", val.index.min(), "->", val.index.max())
print("Test: ", test.index.min(), "->", test.index.max())

# Number of observations
print("\nNumber of observations:")
print("Train:", len(train))
print("Val:  ", len(val))
print("Test: ", len(test))
print("Features:", FEATS, "| NOWCAST =", NOWCAST)


def rolling_forecast(order, fit_y, fit_x, eval_y, eval_x):
    res = ARIMA(fit_y, exog=fit_x, order=order).fit()
    preds = []
    for t in range(len(eval_y)):
        x_t = eval_x.iloc[t:t + 1]
        preds.append(float(res.forecast(steps=1, exog=x_t).iloc[0]))
        res = res.append(eval_y.iloc[t:t + 1], exog=x_t, refit=False)
    return pd.Series(preds, index=eval_y.index)


# -------------------------------------
# 2. Hyperparameter tuning (p, d, q) on the VALIDATION set
#    The test set is not used here.
# -------------------------------------
p_grid = [1, 2, 3, 6, 12]
d_grid = [0, 1]
q_grid = [0, 1, 2]

results = []
for order in itertools.product(p_grid, d_grid, q_grid):
    try:
        pred = rolling_forecast(order, train, X_train, val, X_val)
        rmse = np.sqrt(mean_squared_error(val, pred))
        results.append((order, rmse))
        print(f"ARIMAX{order}  val RMSE = {rmse:,.0f}")
    except Exception as e:
        print(f"ARIMAX{order}  failed: {e}")

best_order, best_rmse = min(results, key=lambda r: r[1])
print("\nBest order:", best_order, "| validation RMSE:", round(best_rmse, 2))

n_steps = len(test)

res = ARIMA(train, exog=X_train, order=best_order).fit()
res = res.append(val, exog=X_val, refit=False)          # validation -> model state

preds = []
for t in range(len(test)):
    x_t = X_test.iloc[t:t + 1]
    preds.append(float(res.forecast(steps=1, exog=x_t).iloc[0]))     # one step ahead
    res = res.append(test.iloc[t:t + 1], exog=x_t, refit=False)      # observed test value -> state

forecast = pd.Series(preds, index=test.index)

print(forecast)
print(forecast.shape)




# -------------------------------------
# 4. Future dates (same as the basic code)
# -------------------------------------
split_date_dt = pd.to_datetime('2024-04-01')

future_dates = pd.date_range(start=split_date_dt,
                             periods=n_steps,
                             freq='MS')
assert (future_dates == test.index).all()

forecast.index = future_dates
forecast.name = 'predicted_arrivals'

print(forecast)
forecast.to_csv("Results/test_predicted_arimax.csv")   # same file format as the basic code


# -------------------------------------
# 5. Metrics
# -------------------------------------
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100
print(f"Test  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

#pd.DataFrame([{"model": "ARIMAX", "order": str(best_order), "MAE": mae, "RMSE": rmse,  "MAPE": mape, "features": ",".join(FEATS), "nowcast": NOWCAST}]
# ).to_csv("arimax_test_metrics.csv", index=False)

with open("Results/best_hparams_arimax.json", "w") as f:
    json.dump({"order": str(best_order)}, f)

