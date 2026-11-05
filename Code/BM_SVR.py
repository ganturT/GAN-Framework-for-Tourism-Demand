import numpy as np
import pandas as pd
import json
from pickle import load

from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# 0. Settings
REFIT_ON_TRAIN_PLUS_VAL = False

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
X_train = np.load("Data/X_train.npy", allow_pickle=True)
y_train = np.load("Data/y_train.npy", allow_pickle=True)
X_val = np.load("Data/X_val.npy", allow_pickle=True)
y_val = np.load("Data/y_val.npy", allow_pickle=True)
X_test = np.load("Data/X_test.npy", allow_pickle=True)
y_test = np.load("Data/y_test.npy", allow_pickle=True)

output_dim = y_train.shape[1]
print("X_train:", X_train.shape, "| X_val:", X_val.shape, "| X_test:", X_test.shape)

# SVR needs 2D inputs and a 1D target
X_tr = X_train.reshape(X_train.shape[0], -1)
y_tr = y_train.ravel()
X_va = X_val.reshape(X_val.shape[0], -1)
y_va = y_val.ravel()


# ---------------------------------------------------------------------------
# 2. Hyperparameter selection on the VALIDATION set
#    Fit on train only, score on validation, pick the lowest validation RMSE.
#    The test set is never used here.
# ---------------------------------------------------------------------------
def tuned_SVR():
    param_grid = {
        "C":       [0.1, 1, 10, 100],
        "epsilon": [0.001, 0.01, 0.1],
        "gamma":   ["scale", 0.001, 0.01, 0.1],
    }

    best_rmse, best_params = np.inf, None
    for params in ParameterGrid(param_grid):
        m = SVR(kernel="rbf", **params).fit(X_tr, y_tr)
        val_rmse = np.sqrt(mean_squared_error(y_va, m.predict(X_va)))
        if val_rmse < best_rmse:
            best_rmse, best_params = val_rmse, params

    print("Best params:", best_params)
    print("Best validation RMSE (scaled):", best_rmse)

    with open("Results/best_hparams_svr.json", "w") as f:
        json.dump({"order": str(best_params)}, f)


    if REFIT_ON_TRAIN_PLUS_VAL:
        X_all = np.vstack([X_tr, X_va])
        y_all = np.concatenate([y_tr, y_va])
        return SVR(kernel="rbf", **best_params).fit(X_all, y_all)
    return SVR(kernel="rbf", **best_params).fit(X_tr, y_tr)


model = tuned_SVR()


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

def compute_metrics(real, predicted):
    mae = mean_absolute_error(real, predicted)
    rmse = np.sqrt(mean_squared_error(real, predicted))
    mape = np.mean(np.abs((real - predicted) / real)) * 100
    return mae, rmse, mape


def evaluate(X_raw, y_raw, index_file):
    X_flat = X_raw.reshape(X_raw.shape[0], -1)
    yhat = model.predict(X_flat).reshape(-1, 1)

    y_scaler = load(open("Data/y_scaler.pkl", "rb"))
    index = pd.to_datetime(np.load(index_file, allow_pickle=True))

    rescaled_real = y_scaler.inverse_transform(y_raw)
    rescaled_pred = y_scaler.inverse_transform(yhat)

    pred_df = pd.DataFrame({"predicted_arrivals": rescaled_pred[:, 0]}, index=index)
    real = pd.Series(rescaled_real[:, 0], index=index)
    return pred_df, compute_metrics(real, pred_df["predicted_arrivals"])


# ---------------------------------------------------------------------------
# 4. Test set evaluation
# ---------------------------------------------------------------------------
pred_df, (mae, rmse, mape) = evaluate(X_test, y_test, "Data/index_test.npy")
pred_df.to_csv("Results/test_predicted_svr.csv")
print(f"Test   MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")