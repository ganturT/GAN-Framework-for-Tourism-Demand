import numpy as np
import pandas as pd
from pickle import load
from sklearn.metrics import mean_absolute_error, mean_squared_error

METRICS = ["MAE", "RMSE", "MAPE"]

# Neural models: Results/test_summary_<prefix>.csv (written by the multi-seed scripts)
NEURAL = {"GAN (BCE)": "gan",
          "GAN (BCE + MSE)": "gan_bce_mse",
          "GAN (WGAN-GP)": "gan_wgan_gp",
          "ANN": "ann", "LSTM": "lstm", "GRU": "gru"}
# Deterministic models: fitted once, metrics computed from the predicted csv
DETERMINISTIC = {"Seasonal naive": "Results/test_predicted_snaive.csv",
                 "SVR": "Results/test_predicted_svr.csv",
                 "ARIMAX": "Results/test_predicted_arimax.csv",
                 "SARIMAX": "Results/test_predicted_sarimax.csv"}

# Row order of Table 2, in groups (a \midrule is placed between groups)
GROUPS = [("GANs (loss functions)", ["GAN (BCE)", "GAN (BCE + MSE)", "GAN (WGAN-GP)"]),
          ("Benchmarks", ["Seasonal naive"]),
          (None, ["SVR", "ARIMAX", "SARIMAX"]),
          (None, ["ANN", "LSTM", "GRU"])]
ROW_ORDER = [n for _, rows in GROUPS for n in rows]


y_test = np.load("Data/y_test.npy", allow_pickle=True).astype("float32")
y_scaler = load(open("Data/y_scaler.pkl", "rb"))   # fitted on the training period only
real = pd.Series(y_scaler.inverse_transform(y_test).ravel(), name="real")


def single_run_metrics(path):
    pred = pd.read_csv(path)["predicted_arrivals"].to_numpy()
    assert len(pred) == len(real), f"{path}: {len(pred)} predictions, {len(real)} test months"
    return {"MAE": mean_absolute_error(real, pred),
            "RMSE": float(np.sqrt(mean_squared_error(real, pred))),
            "MAPE": float(np.mean(np.abs((real - pred) / real)) * 100)}


means, stds = {}, {}

# --- neural models: mean ± std over seeds ---
for name, prefix in NEURAL.items():
    s = pd.read_csv(f"Results/test_summary_{prefix}.csv", index_col=0)   # rows: MAE/RMSE/MAPE, cols: mean/std
    means[name] = {m: s.loc[m, "mean"] for m in METRICS}
    stds[name] = {m: s.loc[m, "std"] for m in METRICS}

# --- deterministic models (incl. seasonal naive): single run, no std ---
for name, path in DETERMINISTIC.items():
    means[name] = single_run_metrics(path)
    stds[name] = {m: None for m in METRICS}

mean_df = pd.DataFrame(means).T.loc[ROW_ORDER, METRICS]
best = mean_df.idxmin()          # best (lowest mean) model for every metric


def fmt(name, m, latex=False):
    mu, sd = mean_df.loc[name, m], stds[name][m]
    pm = r" $\pm$ " if latex else " ± "
    return f"{mu:,.2f}{pm}{sd:,.2f}" if sd is not None else f"{mu:,.2f}"


# ---------- console / CSV ----------
table = pd.DataFrame({m: [fmt(n, m) for n in ROW_ORDER] for m in METRICS}, index=ROW_ORDER)
table.index.name = "Models"
table.columns = ["MAE", "RMSE", "MAPE (%)"]
print(table.to_string())
table.to_csv("Results/table2_mean_std.csv")

# Change of each GAN variant relative to the BCE GAN (for the text)
print("\nChange relative to GAN (BCE):")
for n in ["GAN (BCE + MSE)", "GAN (WGAN-GP)"]:
    d = {m: (mean_df.loc[n, m] / mean_df.loc["GAN (BCE)", m] - 1) * 100 for m in ["MAE", "RMSE"]}
    dm = mean_df.loc[n, "MAPE"] - mean_df.loc["GAN (BCE)", "MAPE"]
    print(f"  {n:<16} MAE {d['MAE']:+.1f}%  RMSE {d['RMSE']:+.1f}%  MAPE {dm:+.2f} points")

# ---------- LaTeX (MDPI style, best mean in bold, grouped rows) ----------
lines = [r"\begin{tabularx}{\fulllength}{lCCC}", r"\toprule",
         r"\textbf{Models} & \textbf{MAE} & \textbf{RMSE} & \textbf{MAPE (\%)} \\", r"\midrule"]
for i, (title, rows) in enumerate(GROUPS):
    if i > 0:
        lines.append(r"\midrule")
    if title:
        lines += [rf"\multicolumn{{4}}{{l}}{{\textit{{{title}}}}} \\", r"\midrule"]
    for n in rows:
        cells = []
        for m in METRICS:
            c = fmt(n, m, latex=True)
            cells.append(rf"\textbf{{{c}}}" if best[m] == n else c)
        lines.append(f"{n} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabularx}",
          r"% Neural models (GAN variants, ANN, LSTM, GRU): mean $\pm$ std over 4 seeds. "
          r"Seasonal naive, SVR, ARIMAX, SARIMAX: deterministic, single run."]
with open("Results/table2_mean_std.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved: Results/table2_mean_std.csv, Results/table2_mean_std.tex")