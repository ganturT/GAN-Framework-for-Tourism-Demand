import os
import gc
import json
import random
import time
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pickle import load
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, LeakyReLU, GRU

# ===================== SEEDS ===============================
TUNE_SEEDS = [10,11,12]      # grid search: every config is run with ALL of these seeds
FINAL_SEEDS = [0, 1, 2, 3]     # final model is trained with each of these seeds (different from TUNE_SEEDS)


def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)



# ===================== SETTINGS ============================
# Full grid used in the paper (Table 1):
LR_GRID = [0.0001, 0.0002, 0.0005, 0.001, 0.0016, 0.003, 0.005]
UNITS_GRID = [16, 32, 64, 128, 256]     # generator units
D_UNITS_GRID = [8, 16, 32, 64, 128]      # discriminator units (Conv1D filters d/2, d, d; Dense d, d/2)
## The selected values are fixed below to reproduce the final results without re-running the search.

 #Selected values (Table 1). Delete these lines to repeat the full grid search.
LR_GRID = [0.0002]
UNITS_GRID = [32]
D_UNITS_GRID = [64]

EPOCHS = 300
BATCH_SIZE = 4
PATIENCE = 30



# Every run is saved here immediately; if the script stops (memory, sleep, Ctrl+C),
# running it again skips the runs already in this file.
TUNING_LOG = "Results/tuning_runs_gan_bce.csv"

# ===================== DATA ================================

X_train = np.load("Data/X_train.npy", allow_pickle=True).astype("float64")
y_train = np.load("Data/y_train.npy", allow_pickle=True).astype("float64")

X_val = np.load("Data/X_val.npy", allow_pickle=True).astype("float64")
y_val = np.load("Data/y_val.npy", allow_pickle=True).astype("float64")

X_test = np.load("Data/X_test.npy", allow_pickle=True).astype("float64")
y_test = np.load("Data/y_test.npy", allow_pickle=True).astype("float64")
y_scaler = load(open("Data/y_scaler.pkl", "rb"))   # fitted on training data only

VOLUME_COL = 0   # column of X that holds the (scaled) tourist volume

# ===================== MODELS ==============================

def build_generator(window_size, horizon, n_features, units):
    """GRU generator: 12-month input window -> next-month forecast."""
    recurrent = [GRU(units, return_sequences=True), GRU(units // 2)]
    head = [Dense(n, activation="relu") for n in (units, max(units // 2, 1))] + [Dense(horizon)]
    return Sequential([tf.keras.Input(shape=(window_size, n_features))] + recurrent + head,
                      name="generator")


def build_discriminator(window_size, d_units=64):
    """Conv1D discriminator: scores a (window_size + 1)-step arrival sequence as real or generated."""
    conv_spec = [(d_units // 2, 3), (d_units, 4), (d_units, 5)]          # (filters, kernel size)
    conv_blocks = [Conv1D(filters, kernel, strides=2, padding="same", activation=LeakyReLU(alpha=0.01))
                   for filters, kernel in conv_spec]
    classifier = [Flatten(),
                  Dense(d_units, use_bias=False), LeakyReLU(),
                  Dense(d_units // 2, use_bias=False, activation="relu"),
                  Dense(1, activation="sigmoid")]
    return Sequential([tf.keras.Input(shape=(window_size + 1, 1))] + conv_blocks + classifier,
                      name="discriminator")

# ===================== METRICS (original scale) ============
def compute_metrics(y_true_scaled, y_pred_scaled):
    y_true = y_scaler.inverse_transform(y_true_scaled).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / y_true)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


# =====================  GAN TRAINER =============
class GANTrainer:

    def __init__(self, generator, discriminator, lr):
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.opt_d = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @staticmethod
    def with_history(x, next_value):
        """[next-month value, past 12 months of arrivals] as a (window + 1, 1) sequence."""
        next_value = tf.cast(tf.reshape(next_value, [-1, tf.shape(next_value)[1], 1]), tf.float64)
        history = tf.cast(x[:, :, VOLUME_COL:VOLUME_COL + 1], tf.float64)
        return tf.concat([next_value,history], axis=1)

    @tf.function
    def step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            forecast = self.generator(x, training=True)
            p_real = self.discriminator(self.with_history(x, y), training=True)
            p_fake = self.discriminator(self.with_history(x, forecast), training=True)

            loss_g = self.bce(tf.ones_like(p_fake), p_fake)
            loss_d = self.bce(tf.ones_like(p_real), p_real) + self.bce(tf.zeros_like(p_fake), p_fake)

        # Compute both gradients from the same forward pass, then update both networks
        updates = [(tape.gradient(loss, net.trainable_variables), net, opt)
                   for loss, net, opt in ((loss_g, self.generator, self.opt_g),
                                          (loss_d, self.discriminator, self.opt_d))]
        for grads, net, opt in updates:
            opt.apply_gradients(zip(grads, net.trainable_variables))
        del tape
        return loss_d, loss_g


def predict(g_model, X):
    return g_model(X, training=False).numpy()


def train_gan(g_model, d_model, lr, epochs, batch_size,
              X_tr, y_tr, X_va, y_va, patience=30, verbose=True):

    trainer = GANTrainer(g_model, d_model, lr)
    adversarial_step = trainer.step

    hist = {"d_loss": [], "g_loss": [], "train_rmse": [], "val_rmse": [], "val_mae": [], "val_mape": []}
    best_val, best_epoch, best_weights, wait = np.inf, 0, None, 0
    ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(batch_size)

    for epoch in range(epochs):
        t0 = time.time()
        d_losses, g_losses = [], []
        for bx, by in ds:
            d_l, g_l = adversarial_step(bx, by)
            d_losses.append(float(d_l))
            g_losses.append(float(g_l))

        tr_m = compute_metrics(y_tr, predict(g_model, X_tr))
        va_m = compute_metrics(y_va, predict(g_model, X_va))
        hist["d_loss"].append(np.mean(d_losses))
        hist["g_loss"].append(np.mean(g_losses))
        hist["train_rmse"].append(tr_m["RMSE"])
        hist["val_rmse"].append(va_m["RMSE"])
        hist["val_mae"].append(va_m["MAE"])
        hist["val_mape"].append(va_m["MAPE"])

        # Select the best epoch on VALIDATION RMSE (never on test)
        if va_m["RMSE"] < best_val:
            best_val, best_epoch, wait = va_m["RMSE"], epoch + 1, 0
            best_weights = g_model.get_weights()
        else:
            wait += 1

        if verbose:
            print(f"epoch {epoch + 1:3d} | d_loss {hist['d_loss'][-1]:.3f} g_loss {hist['g_loss'][-1]:.3f} "
                  f"| train_rmse {tr_m['RMSE']:.0f} val_rmse {va_m['RMSE']:.0f} "
                  f"val_mae {va_m['MAE']:.0f} | {time.time() - t0:.1f}s")

        if patience is not None and wait >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1} (best epoch {best_epoch})")
            break

    # Restore the best-validation generator
    g_model.set_weights(best_weights)
    return hist, best_epoch


def free_memory():
    """Release TensorFlow graphs and Python objects left over from the previous run."""
    tf.keras.backend.clear_session()
    gc.collect()


def run_gan(units, lr, seed, window_size, horizon, n_features, d_units=64):
    free_memory()
    set_seed(seed)
    g_model = build_generator(window_size, horizon, n_features, units=units)
    d_model = build_discriminator(window_size, d_units=d_units)
    hist, best_epoch = train_gan(g_model, d_model, lr, EPOCHS, BATCH_SIZE,
                                 X_train, y_train, X_val, y_val,
                                 patience=PATIENCE, verbose=False)
    del d_model
    return g_model, hist, best_epoch


# ===================== PLOTS / SAVING ======================
def plot_gan_loss_multi_seed(histories, skip=4, filename="Results/training_loss_all_seeds_gan.pdf"):
    _ , axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for seed, h in histories.items():
        epochs = np.arange(1, len(h["d_loss"]) + 1)
        axes[0].plot(epochs[skip:], np.asarray(h["d_loss"])[skip:], linewidth=1.5, label=f"seed {seed}")
        axes[1].plot(epochs[skip:], np.asarray(h["g_loss"])[skip:], linewidth=1.5, label=f"seed {seed}")

    axes[0].set_title("D loss")
    axes[1].set_title("G loss")
    axes[0].legend(loc="lower left")
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def save_test_predictions(test_preds, model_name, index_file="Data/index_test.npy"):

    mean_test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)

    # Dates of the test target months (saved by data_preparation.py)
    test_dates = pd.to_datetime(np.load(index_file, allow_pickle=True))
    assert len(test_dates) == len(mean_test_pred), "test dates and predictions do not match"

    result = pd.DataFrame({
        "": test_dates.strftime("%Y-%m-%d"),
        "seeds_mean_prediction": y_scaler.inverse_transform(mean_test_pred).ravel(),
    })
    result.to_csv(f"Results/test_predicted_meanseed_{model_name}.csv", index=False)
    return result


# ===================== ONE RUN IN A SEPARATE PROCESS =======
# Each training run is executed in a fresh Python process that exits afterwards,
# so all TensorFlow memory is returned to the operating system after every run.
# Memory therefore does not grow, no matter how many runs the grid has.
def _one_run(args):
    units, lr, d_units, seed, final = args
    window_size, n_features, horizon = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    g_model, hist, best_epoch = run_gan(units, lr, seed, window_size, horizon, n_features, d_units=d_units)
    out = {"best_epoch": best_epoch,
           "val": compute_metrics(y_val, predict(g_model, X_val))}
    if final:   # test set is evaluated only in the final runs
        yhat_test = predict(g_model, X_test)
        out.update({"train": compute_metrics(y_train, predict(g_model, X_train)),
                    "test": compute_metrics(y_test, yhat_test),
                    "yhat_test": yhat_test,
                    "hist": {"d_loss": hist["d_loss"], "g_loss": hist["g_loss"]}})
    return out


def run_isolated(units, lr, d_units, seed, final=False):
    ctx = mp.get_context("spawn")           # works on Windows and Linux
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        return pool.apply(_one_run, ((units, lr, d_units, seed, final),))


# ===================== MAIN ================================
if __name__ == "__main__":
    os.makedirs("Results", exist_ok=True)

    # STEP 1: GRID SEARCH over MULTIPLE seeds - test set NOT used
    # Resume: runs already saved in TUNING_LOG are not repeated.
    if os.path.exists(TUNING_LOG):
        done_df = pd.read_csv(TUNING_LOG)
        done = {(round(float(r.lr), 6), int(r.units), int(r.d_units), int(r.seed))
                for r in done_df.itertuples()}
        print(f"Resuming: {len(done)} runs already in {TUNING_LOG}")
    else:
        done = set()

    grid = list(itertools.product(LR_GRID, UNITS_GRID, D_UNITS_GRID))
    total, k = len(grid) * len(TUNE_SEEDS), 0
    for lr, units, d_units in grid:
        for s in TUNE_SEEDS:
            k += 1
            if (round(float(lr), 6), units, d_units, s) in done:
                continue
            t0 = time.time()
            out = run_isolated(units, lr, d_units, s)
            va = out["val"]
            row = {"lr": lr, "units": units, "d_units": d_units, "seed": s, "best_epoch": out["best_epoch"],
                   "val_rmse": round(va["RMSE"]), "val_mae": round(va["MAE"]), "val_mape": round(va["MAPE"], 2)}
            pd.DataFrame([row]).to_csv(TUNING_LOG, mode="a", index=False,
                                       header=not os.path.exists(TUNING_LOG))
            print(f"[{k}/{total}] lr={lr:<7} units={units:<4} d_units={d_units:<4} seed={s:<3} "
                  f"best_epoch={out['best_epoch']:<4} val RMSE={va['RMSE']:.0f}  MAE={va['MAE']:.0f}  "
                  f"MAPE={va['MAPE']:.2f}%  ({time.time() - t0:.0f}s)")

    # Use only the runs of the current grid (the log may contain other runs)
    runs_df = pd.read_csv(TUNING_LOG)
    runs_df = runs_df[runs_df["lr"].round(6).isin([round(x, 6) for x in LR_GRID])
                      & runs_df["units"].isin(UNITS_GRID) & runs_df["d_units"].isin(D_UNITS_GRID)
                      & runs_df["seed"].isin(TUNE_SEEDS)]

    # Aggregate each (lr, units, d_units) config over the tuning seeds
    res = (runs_df.groupby(["lr", "units", "d_units"])
           .agg(val_rmse_mean=("val_rmse", "mean"),
                val_rmse_median=("val_rmse", "median"),
                val_rmse_std=("val_rmse", "std"),
                val_rmse_worst=("val_rmse", "max"),
                val_mape_mean=("val_mape", "mean"),
                best_epoch_mean=("best_epoch", "mean"))
           .reset_index())

    # Selection criterion: mean + std of val RMSE (penalises unstable configs)
    res["score"] = res["val_rmse_mean"] + res["val_rmse_std"]
    res = res.sort_values("score")
    #res.to_csv("Results/tuning_summary_gan.csv", index=False)
    print("\n=== Tuning results (multi-seed, sorted by mean + std of val RMSE) ===")
    print(res.round(1).to_string(index=False))

    # Hyperparameters are now FIXED for all seeds
    BEST_LR = float(res.iloc[0]["lr"])
    BEST_UNITS = int(res.iloc[0]["units"])
    BEST_D_UNITS = int(res.iloc[0]["d_units"])
    with open("Results/best_hparams_gan.json", "w") as f:
        json.dump({"lr": BEST_LR, "units": BEST_UNITS, "d_units": BEST_D_UNITS,
                   "tune_seeds": TUNE_SEEDS}, f)
    print(f"\nFixed hyperparameters: lr={BEST_LR}, units={BEST_UNITS}, d_units={BEST_D_UNITS} "
          f"(selected over seeds {TUNE_SEEDS}, mean val RMSE={res.iloc[0]['val_rmse_mean']:.0f})")

    # STEP 2: FINAL TRAINING - same hyperparameters, different seeds
    # Test set is evaluated ONCE per seed, after all choices are fixed.
    records, test_preds, histories = [], [], {}
    for seed in FINAL_SEEDS:
        out = run_isolated(BEST_UNITS, BEST_LR, BEST_D_UNITS, seed, final=True)
        tr, va, te = out["train"], out["val"], out["test"]
        test_preds.append(out["yhat_test"])
        histories[seed] = out["hist"]
        records.append({"seed": seed, "best_epoch": out["best_epoch"],
                        "train_RMSE": tr["RMSE"], "train_MAE": tr["MAE"], "train_MAPE": tr["MAPE"],
                        "val_RMSE": va["RMSE"], "val_MAE": va["MAE"], "val_MAPE": va["MAPE"],
                        "test_RMSE": te["RMSE"], "test_MAE": te["MAE"], "test_MAPE": te["MAPE"]})
        print(f"seed={seed} best_epoch={out['best_epoch']:<4} "
              f"TEST RMSE={te['RMSE']:.0f}  MAE={te['MAE']:.0f}  MAPE={te['MAPE']:.2f}%")

    per_seed = pd.DataFrame(records)
    #per_seed.to_csv("Results/per_seed_gan.csv", index=False)

    # mean +- std over seeds (sample std, ddof=1)
    print("\n=== Test results over seeds (mean ± std) ===")
    summary = {}
    for m in ["MAE", "RMSE", "MAPE"]:
        mean, std = per_seed[f"test_{m}"].mean(), per_seed[f"test_{m}"].std(ddof=1)
        summary[m] = {"mean": mean, "std": std}
        print(f"{m:<5}: {mean:.2f} ± {std:.2f}")
    pd.DataFrame(summary).T.to_csv("Results/test_summary_gan.csv")

    # D loss | G loss over all final seeds
    plot_gan_loss_multi_seed(histories)

    # Save test mean prediction over seeds (original scale)
    save_test_predictions(test_preds, "gan")