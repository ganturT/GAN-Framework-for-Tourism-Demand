import os
import json
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pickle import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ===================== SEEDS (same as GAN) =================
TUNE_SEEDS = [10, 11, 12]      # grid search: every config is run with ALL of these seeds
FINAL_SEEDS = [0, 1, 2, 3]     # final model is trained with each of these seeds (different from TUNE_SEEDS)


def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)

# ===================== SETTINGS (same as GAN) ==============
BATCH_SIZE = 4
N_EPOCH = 300          
PATIENCE = 30         

LR_GRID = [0.0002, 0.0005, 0.001, 0.0016, 0.003, 0.005]
UNITS_GRID = [ 64, 128, 256,512,1024]          

# Selected values (Table 1). Delete these lines to repeat the full grid search
LR_GRID = [0.0016]
UNITS_GRID = [512]   

# ===================== DATA ================================
X_train_raw = np.load("Data/X_train.npy", allow_pickle=True).astype("float32")
y_train = np.load("Data/y_train.npy", allow_pickle=True).astype("float32")
X_val_raw = np.load("Data/X_val.npy", allow_pickle=True).astype("float32")
y_val = np.load("Data/y_val.npy", allow_pickle=True).astype("float32")
X_test_raw = np.load("Data/X_test.npy", allow_pickle=True).astype("float32")
y_test = np.load("Data/y_test.npy", allow_pickle=True).astype("float32")
y_scaler = load(open("Data/y_scaler.pkl", "rb"))   # fitted on the training period only

# The ANN takes a flat vector: (samples, input_dim * feature_size)
X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
X_val = X_val_raw.reshape(X_val_raw.shape[0], -1)
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
print("input_dim:", input_dim, "| output_dim:", output_dim)


# ===================== MODEL ===============================
def build_ann(units, lr):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(units, activation="relu"))
    model.add(Dense(max(units // 2, 8), activation="relu"))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss="mse")
    return model


def fit_ann(units, lr, seed, verbose=0):
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_ann(units, lr)
    # val_loss is MSE on the scaled target; since the inverse scaling is linear,
    # minimising it is equivalent to minimising the validation RMSE (original scale).
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val), callbacks=[es],
                        verbose=verbose, shuffle=False)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return model, history, best_epoch


# ===================== METRICS (original scale) ============
def compute_metrics(y_true_scaled, y_pred_scaled):
    y_true = y_scaler.inverse_transform(y_true_scaled).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
    err = y_true - y_pred
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "MAPE": float(np.mean(np.abs(err / y_true)) * 100)}


# =====================================================================
# STEP 1: GRID SEARCH over MULTIPLE seeds - test set NOT used
# =====================================================================
runs = []   # one row = one (lr, units, seed) run

for lr, units in itertools.product(LR_GRID, UNITS_GRID):
    for s in TUNE_SEEDS:
        model, history, best_epoch = fit_ann(units, lr, s)
        va = compute_metrics(y_val, model.predict(X_val, verbose=0))
        runs.append({"lr": lr, "units": units, "seed": s, "best_epoch": best_epoch,
                     "val_rmse": va["RMSE"], "val_mae": va["MAE"], "val_mape": va["MAPE"]})
        print(f"lr={lr:<7} units={units:<4} seed={s:<3} best_epoch={best_epoch:<4} "
              f"val RMSE={va['RMSE']:.0f}  MAE={va['MAE']:.0f}  MAPE={va['MAPE']:.2f}%")

runs_df = pd.DataFrame(runs)
#runs_df.to_csv("ann_tuning_all_runs.csv", index=False)

# Aggregate each (lr, units) config over the tuning seeds
res = (runs_df.groupby(["lr", "units"])
       .agg(val_rmse_mean=("val_rmse", "mean"),
            val_rmse_median=("val_rmse", "median"),
            val_rmse_std=("val_rmse", "std"),
            val_rmse_worst=("val_rmse", "max"),
            val_mape_mean=("val_mape", "mean"),
            best_epoch_mean=("best_epoch", "mean"))
       .reset_index())

# Selection criterion: mean + std of val RMSE (SAME as GAN)
res["score"] = res["val_rmse_mean"] + res["val_rmse_std"]
res = res.sort_values("score")
#res.to_csv("ann_tuning_results.csv", index=False)
print("\n=== Tuning results (multi-seed, sorted by mean + std of val RMSE) ===")
print(res.round(4).to_string(index=False))

# Hyperparameters are now FIXED for all seeds
BEST_LR = float(res.iloc[0]["lr"])
BEST_UNITS = int(res.iloc[0]["units"])
with open("Results/best_hparams_ann.json", "w") as f:
    json.dump({"lr": BEST_LR, "units": BEST_UNITS, "tune_seeds": TUNE_SEEDS}, f)
print(f"\nFixed hyperparameters: lr={BEST_LR}, units={BEST_UNITS} "
      f"(selected over seeds {TUNE_SEEDS}, mean val RMSE={res.iloc[0]['val_rmse_mean']:.0f})")

# =====================================================================
# STEP 2: FINAL TRAINING - same hyperparameters, different seeds
# Test set is evaluated ONCE per seed, after all choices are fixed.
# =====================================================================
records = []
test_preds = []        # scaled test predictions of every seed
histories = {}

for seed in FINAL_SEEDS:
    model, history, best_epoch = fit_ann(BEST_UNITS, BEST_LR, seed)
    tr = compute_metrics(y_train, model.predict(X_train, verbose=0))
    va = compute_metrics(y_val, model.predict(X_val, verbose=0))
    yhat_test = model.predict(X_test, verbose=0)
    te = compute_metrics(y_test, yhat_test)

    test_preds.append(yhat_test)
    histories[seed] = (history.history, best_epoch)
    #model.save(f"ANN_seed{seed}.keras")
    #pd.DataFrame(history.history).to_csv(f"ann_training_history_seed{seed}.csv", index=False)

    records.append({"seed": seed, "best_epoch": best_epoch,
                    "train_RMSE": tr["RMSE"], "train_MAE": tr["MAE"], "train_MAPE": tr["MAPE"],
                    "val_RMSE": va["RMSE"], "val_MAE": va["MAE"], "val_MAPE": va["MAPE"],
                    "test_RMSE": te["RMSE"], "test_MAE": te["MAE"], "test_MAPE": te["MAPE"]})
    print(f"seed={seed} best_epoch={best_epoch:<4} "
          f"TEST RMSE={te['RMSE']:.0f}  MAE={te['MAE']:.0f}  MAPE={te['MAPE']:.2f}%")

per_seed = pd.DataFrame(records)
#per_seed.to_csv("ann_per_seed_results.csv", index=False)

# mean +- std over seeds (sample std, ddof=1)
print("\n=== Test results over seeds (mean ± std) ===")
summary = {}
for m in ["MAE", "RMSE", "MAPE"]:
    mean, std = per_seed[f"test_{m}"].mean(), per_seed[f"test_{m}"].std(ddof=1)
    summary[m] = {"mean": mean, "std": std}
    print(f"{m:<5}: {mean:.2f} ± {std:.2f}")
pd.DataFrame(summary).T.to_csv("Results/test_summary_ann.csv")

# ===================== TRAINING LOSS PLOT (all seeds 0,1,2,3) =========
def plot_training_loss_all_seeds(histories, filename="Results/training_loss_all_seeds_ann.pdf"):
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for seed, (hist, best_epoch) in histories.items():
        c = colors.get(seed)
        epochs = np.arange(1, len(hist["loss"]) + 1)

        # left panel: train loss, right panel: validation loss (same color per seed)
        axes[0].plot(epochs, hist["loss"], color=c, linewidth=2, label=f"Seed {seed}")
        axes[1].plot(epochs, hist["val_loss"], color=c, linewidth=2, label=f"Seed {seed}")
        # mark the epoch whose weights were restored (best validation epoch)
        axes[1].scatter(best_epoch, hist["val_loss"][best_epoch - 1],
                        color=c, edgecolor="black", zorder=5, s=50)

    axes[0].set_title("Train loss")
    axes[1].set_title("Validation loss (● = best epoch)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE, scaled)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("(a) ANN", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

plot_training_loss_all_seeds(histories)


# ===================== SAVE TEST MEAN PREDICTION  OVER SEEDS BY REAL VALUE  =====
def save_predictions(yhat, index_file):
    index = np.load(index_file, allow_pickle=True)
    pred = y_scaler.inverse_transform(yhat)
    return pd.DataFrame({'seeds_mean_prediction': pred[:, 0]}, index=index[:len(pred)])


mean_test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
predict_result = save_predictions(mean_test_pred, "Data/index_test.npy")
predict_result.to_csv('Results/test_predicted_meanseed_ann.csv')




