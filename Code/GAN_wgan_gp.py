import os
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

MODEL_NAME = "gan_wgan_gp"

# ===================== SEEDS ===============================
TUNE_SEEDS = [10, 11, 12]      # grid search: every config is run with ALL of these seeds
FINAL_SEEDS = [0, 1, 2, 3]     # final model is trained with each of these seeds (different from TUNE_SEEDS)


def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)


# ===================== SETTINGS ============================
# Same grid as the BCE GAN (Table 1)
LR_GRID = [0.0001, 0.0002, 0.0005, 0.001, 0.0016, 0.003, 0.005]
UNITS_GRID = [16, 32, 64, 128, 256]     # generator units
D_UNITS_GRID = [8, 16, 32, 64, 128]      # discriminator units (Conv1D filters d/2, d, d; Dense d, d/2)
# The selected values are fixed below to reproduce the final results without re-running the search.

# Selected values (Table 1). Delete these lines to repeat the full grid search.
LR_GRID = [0.001]
UNITS_GRID = [64]
D_UNITS_GRID = [32]

N_CRITIC_GRID = [5]                 # critic updates per generator update (5 = standard value);
                                    # use [1, 5] to tune it as well (doubles the number of runs)
GP_WEIGHT = 10.0                    # gradient-penalty weight (standard value)

EPOCHS = 300
BATCH_SIZE = 4
PATIENCE = 30

# Every run is saved here immediately; if the script stops (memory, sleep, Ctrl+C),
# running it again skips the runs already in this file.
TUNING_LOG = f"Results/tuning_runs_{MODEL_NAME}.csv"

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

    recurrent = [GRU(units, return_sequences=True), GRU(units // 2)]
    head = [Dense(n, activation="relu") for n in (units, max(units // 2, 1))] + [Dense(horizon)]
    return Sequential([tf.keras.Input(shape=(window_size, n_features))] + recurrent + head,
                      name="generator")


def build_critic(window_size, d_units=64):
    conv_spec = [(d_units // 2, 3), (d_units, 4), (d_units, 5)]          # (filters, kernel size)
    conv_blocks = [Conv1D(filters, kernel, strides=2, padding="same", activation=LeakyReLU(alpha=0.01))
                   for filters, kernel in conv_spec]
    head = [Flatten(),
            Dense(d_units, use_bias=False), LeakyReLU(),
            Dense(d_units // 2, use_bias=False, activation="relu"),
            Dense(1)]                                                     # linear output
    return Sequential([tf.keras.Input(shape=(window_size + 1, 1))] + conv_blocks + head,
                      name="critic")


# ===================== METRICS (original scale) ============
def compute_metrics(y_true_scaled, y_pred_scaled):
    y_true = y_scaler.inverse_transform(y_true_scaled).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
    err = y_true - y_pred
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "MAPE": float(np.mean(np.abs(err / y_true)) * 100)}


def predict(g_model, X):
    return g_model(X, training=False).numpy()


# ===================== TRAINER =============================
class WGANGPTrainer:
    def __init__(self, generator, critic, lr, n_critic):
        self.generator = generator
        self.critic = critic
        self.n_critic = n_critic
        # Adam settings recommended for WGAN-GP
        self.opt_g = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9, clipnorm=1.0)
        self.opt_c = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9, clipnorm=1.0)

    @staticmethod
    def with_history(x, next_value):
        """[next-month value, past 12 months of arrivals] as a (window + 1, 1) sequence."""
        next_value = tf.cast(tf.reshape(next_value, [-1, tf.shape(next_value)[1], 1]), tf.float64)
        history = tf.cast(x[:, :, VOLUME_COL:VOLUME_COL + 1], tf.float64)
        return tf.concat([next_value, history], axis=1)

    @tf.function
    def critic_step(self, x, y):
        forecast = self.generator(x, training=False)
        # float32 to match the critic's output dtype (otherwise the penalty term is float64)
        real_seq = tf.cast(self.with_history(x, y), tf.float32)
        fake_seq = tf.cast(self.with_history(x, forecast), tf.float32)
        with tf.GradientTape() as tape:
            c_real = self.critic(real_seq, training=True)
            c_fake = self.critic(fake_seq, training=True)
            # gradient penalty on random points between real and fake sequences
            eps = tf.random.uniform([tf.shape(real_seq)[0], 1, 1], 0.0, 1.0, dtype=real_seq.dtype)
            mixed = eps * real_seq + (1.0 - eps) * fake_seq
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(mixed)
                c_mixed = self.critic(mixed, training=True)
            grads = gp_tape.gradient(c_mixed, mixed)
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-12)
            gp = tf.reduce_mean(tf.square(norm - 1.0))
            loss_c = tf.reduce_mean(c_fake) - tf.reduce_mean(c_real) + tf.cast(GP_WEIGHT, gp.dtype) * gp
        self.opt_c.apply_gradients(zip(tape.gradient(loss_c, self.critic.trainable_variables),
                                       self.critic.trainable_variables))
        return loss_c

    @tf.function
    def generator_step(self, x):
        with tf.GradientTape() as tape:
            forecast = self.generator(x, training=True)
            loss_g = -tf.reduce_mean(self.critic(self.with_history(x, forecast), training=True))
        self.opt_g.apply_gradients(zip(tape.gradient(loss_g, self.generator.trainable_variables),
                                       self.generator.trainable_variables))
        return loss_g

    def step(self, x, y):
        for _ in range(self.n_critic):
            loss_c = self.critic_step(x, y)
        return loss_c, self.generator_step(x)


def train_gan(trainer, X_tr, y_tr, X_va, y_va):
    g_model = trainer.generator
    hist = {"c_loss": [], "g_loss": [], "val_rmse": []}
    best_val, best_epoch, best_weights, wait = np.inf, 0, None, 0
    ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(BATCH_SIZE)
    for epoch in range(EPOCHS):
        c_losses, g_losses = [], []
        for bx, by in ds:
            c_l, g_l = trainer.step(bx, by)
            c_losses.append(float(c_l))
            g_losses.append(float(g_l))
        va = compute_metrics(y_va, predict(g_model, X_va))
        hist["c_loss"].append(np.mean(c_losses))
        hist["g_loss"].append(np.mean(g_losses))
        hist["val_rmse"].append(va["RMSE"])
        # Select the best epoch on VALIDATION RMSE (never on test)
        if va["RMSE"] < best_val:
            best_val, best_epoch, wait = va["RMSE"], epoch + 1, 0
            best_weights = g_model.get_weights()
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    g_model.set_weights(best_weights)   # restore the best-validation generator
    return hist, best_epoch


def run_gan(lr, n_critic, units, d_units, seed, shapes):
    tf.keras.backend.clear_session()
    set_seed(seed)
    window_size, horizon, n_features = shapes
    g = build_generator(window_size, horizon, n_features, units)
    c = build_critic(window_size, d_units)
    hist, best_epoch = train_gan(WGANGPTrainer(g, c, lr, n_critic), X_train, y_train, X_val, y_val)
    return g, hist, best_epoch


# ===================== ONE RUN IN A SEPARATE PROCESS =======
# Each training run is executed in a fresh Python process that exits afterwards,
# so all TensorFlow memory is returned to the operating system after every run.
def _one_run(args):
    lr, n_critic, units, d_units, seed, final = args
    shapes = (X_train.shape[1], y_train.shape[1], X_train.shape[2])
    g_model, hist, best_epoch = run_gan(lr, n_critic, units, d_units, seed, shapes)
    out = {"best_epoch": best_epoch,
           "val": compute_metrics(y_val, predict(g_model, X_val))}
    if final:   # test set is evaluated only in the final runs
        yhat_test = predict(g_model, X_test)
        out.update({"train": compute_metrics(y_train, predict(g_model, X_train)),
                    "test": compute_metrics(y_test, yhat_test),
                    "yhat_test": yhat_test,
                    "hist": {"c_loss": hist["c_loss"], "g_loss": hist["g_loss"]}})
    return out


def run_isolated(lr, n_critic, units, d_units, seed, final=False):
    ctx = mp.get_context("spawn")           # works on Windows and Linux
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        return pool.apply(_one_run, ((lr, n_critic, units, d_units, seed, final),))


# ===================== PLOTS / SAVING ======================
def plot_loss_multi_seed(histories, filename=f"Results/training_loss_all_seeds_{MODEL_NAME}.pdf"):
    _, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for s, h in histories.items():
        ep = np.arange(1, len(h["c_loss"]) + 1)
        axes[0].plot(ep, h["c_loss"], linewidth=1.5, label=f"seed {s}")
        axes[1].plot(ep, h["g_loss"], linewidth=1.5, label=f"seed {s}")
    axes[0].set_title("Critic loss")
    axes[1].set_title("G loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def save_test_predictions(test_preds, model_name, index_file="Data/index_test.npy"):
    mean_test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    test_dates = pd.to_datetime(np.load(index_file, allow_pickle=True))
    assert len(test_dates) == len(mean_test_pred), "test dates and predictions do not match"
    result = pd.DataFrame({
        "": test_dates.strftime("%Y-%m-%d"),
        "seeds_mean_prediction": y_scaler.inverse_transform(mean_test_pred).ravel(),
    })
    result.to_csv(f"Results/test_predicted_meanseed_{model_name}.csv", index=False)
    return result


# ===================== MAIN ================================
if __name__ == "__main__":
    os.makedirs("Results", exist_ok=True)
    KEY = ["lr", "n_critic", "units", "d_units"]

    # STEP 1: GRID SEARCH over MULTIPLE seeds - test set NOT used
    # Resume: runs already saved in TUNING_LOG are not repeated.
    if os.path.exists(TUNING_LOG):
        done_df = pd.read_csv(TUNING_LOG)
        done = {(round(float(r.lr), 6), int(r.n_critic), int(r.units), int(r.d_units), int(r.seed))
                for r in done_df.itertuples()}
        print(f"Resuming: {len(done)} runs already in {TUNING_LOG}")
    else:
        done = set()

    grid = list(itertools.product(LR_GRID, N_CRITIC_GRID, UNITS_GRID, D_UNITS_GRID))
    total, k = len(grid) * len(TUNE_SEEDS), 0
    for lr, n, units, d_units in grid:
        for s in TUNE_SEEDS:
            k += 1
            if (round(float(lr), 6), n, units, d_units, s) in done:
                continue
            t0 = time.time()
            out = run_isolated(lr, n, units, d_units, s)
            va = out["val"]
            row = {"lr": lr, "n_critic": n, "units": units, "d_units": d_units, "seed": s,
                   "best_epoch": out["best_epoch"], "val_rmse": round(va["RMSE"]),
                   "val_mae": round(va["MAE"]), "val_mape": round(va["MAPE"], 2)}
            pd.DataFrame([row]).to_csv(TUNING_LOG, mode="a", index=False,
                                       header=not os.path.exists(TUNING_LOG))
            print(f"[{k}/{total}] lr={lr:<7} n_critic={n} units={units:<4} d_units={d_units:<4} seed={s:<3} "
                  f"best_epoch={out['best_epoch']:<4} val RMSE={va['RMSE']:.0f}  MAE={va['MAE']:.0f}  "
                  f"MAPE={va['MAPE']:.2f}%  ({time.time() - t0:.0f}s)")

    # Use only the runs of the current grid (the log may contain other runs)
    runs_df = pd.read_csv(TUNING_LOG)
    runs_df = runs_df[runs_df["lr"].round(6).isin([round(x, 6) for x in LR_GRID])
                      & runs_df["n_critic"].isin(N_CRITIC_GRID)
                      & runs_df["units"].isin(UNITS_GRID) & runs_df["d_units"].isin(D_UNITS_GRID)
                      & runs_df["seed"].isin(TUNE_SEEDS)]

    # Aggregate each configuration over the tuning seeds
    res = (runs_df.groupby(KEY)
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
    #res.to_csv(f"Results/tuning_summary_{MODEL_NAME}.csv", index=False)
    print("\n=== Tuning results (multi-seed, sorted by mean + std of val RMSE) ===")
    print(res.round(1).to_string(index=False))

    # Hyperparameters are now FIXED for all seeds
    best = res.iloc[0]
    BEST_LR, BEST_N = float(best["lr"]), int(best["n_critic"])
    BEST_UNITS, BEST_D_UNITS = int(best["units"]), int(best["d_units"])
    with open(f"Results/best_hparams_{MODEL_NAME}.json", "w") as f:
        json.dump({"lr": BEST_LR, "n_critic": BEST_N, "gp_weight": GP_WEIGHT,
                   "units": BEST_UNITS, "d_units": BEST_D_UNITS, "tune_seeds": TUNE_SEEDS}, f)
    print(f"\nFixed hyperparameters: lr={BEST_LR}, n_critic={BEST_N}, units={BEST_UNITS}, "
          f"d_units={BEST_D_UNITS} (selected over seeds {TUNE_SEEDS}, "
          f"mean val RMSE={best['val_rmse_mean']:.0f})")

    # STEP 2: FINAL TRAINING - same hyperparameters, different seeds
    # Test set is evaluated ONCE per seed, after all choices are fixed.
    records, test_preds, histories = [], [], {}
    for seed in FINAL_SEEDS:
        out = run_isolated(BEST_LR, BEST_N, BEST_UNITS, BEST_D_UNITS, seed, final=True)
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
    #per_seed.to_csv(f"Results/per_seed_{MODEL_NAME}.csv", index=False)

    # mean +- std over seeds (sample std, ddof=1)
    print("\n=== Test results over seeds (mean ± std) ===")
    summary = {}
    for m in ["MAE", "RMSE", "MAPE"]:
        mean, std = per_seed[f"test_{m}"].mean(), per_seed[f"test_{m}"].std(ddof=1)
        summary[m] = {"mean": mean, "std": std}
        print(f"{m:<5}: {mean:.2f} ± {std:.2f}")
    pd.DataFrame(summary).T.to_csv(f"Results/test_summary_{MODEL_NAME}.csv")

    # Critic loss | G loss over all final seeds
    plot_loss_multi_seed(histories)

    # Save test mean prediction over seeds (original scale; used by the DM test)
    save_test_predictions(test_preds, MODEL_NAME)