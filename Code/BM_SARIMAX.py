import warnings
import itertools
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

FEATS = ["search","covid"]
NOWCAST = True      # True: this month's search (lag 0); False: last month's search
S = 12              # seasonal period (months)

df = pd.read_csv('Data/main_dataset.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date').asfreq('MS')

# Monthly series (target) and exogenous features
series = df['volume'].astype(float)
exog = df[FEATS].astype(float).copy()

if not NOWCAST:
    exog['search'] = exog['search'].shift(1)          # last month's search
    ok = exog.notna().all(axis=1)
    series, exog = series[ok], exog[ok]

assert series.isna().sum() == 0, "Missing months in volume"
assert exog.isna().sum().sum() == 0, "Missing values in exogenous features"

# Split dates (same as ARIMAX and the deep learning models)
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

print("Train:", train.index.min(), "->", train.index.max())
print("Val:  ", val.index.min(), "->", val.index.max())
print("Test: ", test.index.min(), "->", test.index.max())
print("\nNumber of observations:")
print("Train:", len(train))
print("Val:  ", len(val))
print("Test: ", len(test))
print("Features:", FEATS, "| NOWCAST =", NOWCAST, "| seasonal period =", S)


# -------------------------------------
# 1. Rolling one-step-ahead forecast
#    Parameters are estimated once on the fit data; after each forecast the
#    actual value AND its features are added to the model state (no re-estimation).
# -------------------------------------
def rolling_forecast(order, seasonal_order, fit_y, fit_x, eval_y, eval_x):
    res = ARIMA(fit_y, exog=fit_x, order=order, seasonal_order=seasonal_order).fit()
    preds = []
    for t in range(len(eval_y)):
        x_t = eval_x.iloc[t:t + 1]
        preds.append(float(res.forecast(steps=1, exog=x_t).iloc[0]))
        res = res.append(eval_y.iloc[t:t + 1], exog=x_t, refit=False)
    return pd.Series(preds, index=eval_y.index)


# -------------------------------------
# 2. Hyperparameter tuning (p, d, q)(P, D, Q, 12) on the VALIDATION set
#    The test set is not used here.
# -------------------------------------
p_grid = [0, 1, 2]
d_grid = [0, 1]
q_grid = [0, 1, 2]
P_grid = [0, 1]
D_grid = [0, 1]
Q_grid = [0, 1]

results = []
for p, d, q, P, D, Q in itertools.product(p_grid, d_grid, q_grid, P_grid, D_grid, Q_grid):
    if p == q == P == Q == 0:
        continue                                     # no AR / MA term at all
    order, seasonal_order = (p, d, q), (P, D, Q, S)
    try:
        pred = rolling_forecast(order, seasonal_order, train, X_train, val, X_val)
        rmse = np.sqrt(mean_squared_error(val, pred))
        results.append((order, seasonal_order, rmse))
        print(f"SARIMAX{order}x{seasonal_order}  val RMSE = {rmse:,.0f}")
    except Exception as e:
        print(f"SARIMAX{order}x{seasonal_order}  failed: {e}")

best_order, best_seasonal, best_rmse = min(results, key=lambda r: r[2])
print("\nBest order:", best_order, "x", best_seasonal, "| validation RMSE:", round(best_rmse, 2))
#pd.DataFrame([(str(o), str(so), r) for o, so, r in sorted(results, key=lambda r: r[2])],
#             columns=["order", "seasonal_order", "val_RMSE"]).to_csv(   "Results/sarimax_tuning_results.csv", index=False)


# -------------------------------------
# 3. Final model: parameters estimated ONCE on train only.
#    Validation, then each observed test value (with its features), is appended
#    to the model state without re-estimating the parameters.
# -------------------------------------
n_steps = len(test)

res = ARIMA(train, exog=X_train, order=best_order, seasonal_order=best_seasonal).fit()
#print(res.summary())
res = res.append(val, exog=X_val, refit=False)          # validation -> model state

preds = []
for t in range(len(test)):
    x_t = X_test.iloc[t:t + 1]
    preds.append(float(res.forecast(steps=1, exog=x_t).iloc[0]))     # one step ahead
    res = res.append(test.iloc[t:t + 1], exog=x_t, refit=False)      # observed test value -> state

forecast = pd.Series(preds, index=test.index)


# -------------------------------------
# 4. Future dates (same as the ARIMAX code)
# -------------------------------------
split_date_dt = pd.to_datetime('2024-04-01')
future_dates = pd.date_range(start=split_date_dt, periods=n_steps, freq='MS')
assert (future_dates == test.index).all()

forecast.index = future_dates
forecast.name = 'predicted_arrivals'
print(forecast)
forecast.to_csv("Results/test_predicted_sarimax.csv")   # same file format as ARIMAX


# -------------------------------------
# 5. Metrics
# -------------------------------------
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true.values - y_pred.values) / y_true.values)) * 100
    return mae, rmse, mape


mae, rmse, mape = metrics(test, forecast)
print(f"SARIMAX test  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

with open("Results/best_hparams_sarimax.json", "w") as f:
    json.dump({"order": str(best_order), "seasonal_order": str(best_seasonal),
               "features": FEATS, "nowcast": NOWCAST, "val_RMSE": float(best_rmse)}, f)


