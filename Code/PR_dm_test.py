import itertools
import numpy as np
import pandas as pd
from pickle import load
from scipy import stats


H = 1                       # forecast horizon (one-step-ahead forecasts)

# GAN variants only (seed-averaged test forecasts written by the GAN scripts)
FILES = {
    "GAN (BCE)":       ("Results/test_predicted_meanseed_gan.csv", "seeds_mean_prediction"),
    "GAN (BCE + MSE)": ("Results/test_predicted_meanseed_gan_bce_mse.csv", "seeds_mean_prediction"),
    "GAN (WGAN-GP)":   ("Results/test_predicted_meanseed_gan_wgan_gp.csv", "seeds_mean_prediction"),
}
# All pairs of loss functions: (model 1, model 2)
PAIRS = list(itertools.combinations(FILES, 2))
# -> (BCE, BCE + MSE), (BCE, WGAN-GP), (BCE + MSE, WGAN-GP)

# Actual test values (same as table2_results.py)
y_test = np.load("Data/y_test.npy", allow_pickle=True).astype("float32")
y_scaler = load(open("Data/y_scaler.pkl", "rb"))
actual = y_scaler.inverse_transform(y_test).ravel().astype(float)


def dm_test(actual, pred1, pred2, h=1, loss="squared"):
    """Returns (DM statistic, two-sided p, one-sided p for 'model 1 is more accurate').
    A negative statistic means model 1 has the smaller loss."""
    e1, e2 = actual - pred1, actual - pred2
    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    T = len(d)
    d_bar = d.mean()
    # long-run variance of d: autocovariances up to lag h-1
    gamma = [np.sum((d[k:] - d_bar) * (d[:T - k] - d_bar)) / T for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    dm = d_bar / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold small-sample correction, compared with t(T-1)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_star = dm * hln
    p_two = 2 * stats.t.sf(abs(dm_star), df=T - 1)
    p_one = stats.t.cdf(dm_star, df=T - 1)          # H1: model 1 more accurate
    return dm_star, p_two, p_one


# ---------------------------------------------------------------------
# Load forecasts
# ---------------------------------------------------------------------
preds = {}
for name, (path, column) in FILES.items():
    df = pd.read_csv(path)
    assert column in df.columns, f"{name}: column '{column}' not found, available: {list(df.columns)}"
    p = df[column].to_numpy(dtype=float)
    assert len(p) == len(actual), f"{name}: {len(p)} forecasts vs {len(actual)} actual values"
    preds[name] = p

# Sanity check: accuracy of the seed-averaged forecasts used in the test
print("Accuracy of the forecasts used in the DM test:")
for name, p in preds.items():
    err = actual - p
    print(f"  {name:<16} MAE {np.mean(np.abs(err)):>10,.2f}  "
          f"RMSE {np.sqrt(np.mean(err ** 2)):>10,.2f}  "
          f"MAPE {np.mean(np.abs(err / actual)) * 100:6.2f}%")
print()

# ---------------------------------------------------------------------
# DM tests between the loss functions
# ---------------------------------------------------------------------
rows, raw = [], {}
for m1, m2 in PAIRS:
    for loss in ["squared", "absolute"]:
        dm, p_two, p_one = dm_test(actual, preds[m1], preds[m2], h=H, loss=loss)
        raw[(m1, m2, loss)] = (dm, p_two, p_one)
        rows.append({"comparison": f"{m1} vs {m2}", "loss": loss,
                     "DM": round(dm, 3), "p_two_sided": round(p_two, 4),
                     "p_one_sided": round(p_one, 4)})

res = pd.DataFrame(rows)
print(f"Test period: {len(actual)} months | horizon h = {H}")
print("DM < 0: the first model has the smaller loss. p_one_sided tests H1: the first model is more accurate.\n")
print(res.to_string(index=False))
res.to_csv("Results/table_dm_test_gans.csv", index=False)
print("\nSaved: Results/table_dm_test_gans.csv")


# ---------------------------------------------------------------------
# LaTeX table of the DM test between the loss functions
# ---------------------------------------------------------------------
def fmt_dm(x):
    return f"${x:.3f}$"          # in math mode, so the minus sign is typeset correctly


def fmt_p(p):
    s = "$<$0.001" if p < 0.001 else f"{p:.3f}"
    return rf"\textbf{{{s}}}" if p < 0.05 else s


def short(name):
    return name.replace("GAN (", "").replace(")", "")      # "GAN (BCE + MSE)" -> "BCE + MSE"


lines = [
    r"\begin{table}[H]",
    r"\caption{Diebold--Mariano test between the loss functions of the GAN on the test set ($h = " + str(H) +
    r"$, Harvey--Leybourne--Newbold correction). A negative statistic indicates that the first loss "
    r"function in each comparison has the smaller loss; $p$-values are two-sided and values below 0.05 "
    r"are shown in bold. The forecasts averaged over four seeds are used for each GAN variant.\label{tab:dm}}",
    r"\begin{adjustwidth}{-\extralength}{0cm}",
    r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
    r"\begin{tabularx}{\fulllength}{lCCCC}",
    r"\toprule",
    r" & \multicolumn{2}{c}{\textbf{Squared Loss}} & \multicolumn{2}{c}{\textbf{Absolute Loss}} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"\textbf{Comparison} & \textbf{DM} & \textbf{\boldmath$p$} & \textbf{DM} & \textbf{\boldmath$p$} \\",
    r"\midrule",
]
for m1, m2 in PAIRS:
    dm_sq, p_sq, _ = raw[(m1, m2, "squared")]
    dm_ab, p_ab, _ = raw[(m1, m2, "absolute")]
    lines.append(f"{short(m1)} vs. {short(m2)} & {fmt_dm(dm_sq)} & {fmt_p(p_sq)} & "
                 f"{fmt_dm(dm_ab)} & {fmt_p(p_ab)} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"]

with open("Results/table_dm_test_gans.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("Saved: Results/table_dm_test_gans.tex")
print("\n" + "\n".join(lines))