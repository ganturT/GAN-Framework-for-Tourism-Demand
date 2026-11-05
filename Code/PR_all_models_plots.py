import pandas as pd
import matplotlib.pyplot as plt

# ===================== SETTINGS ============================

VAL_START = pd.Timestamp("2023-04-01")
TEST_START = pd.Timestamp("2024-04-01")

TRAIN_COLOR = "tab:blue"
VAL_COLOR = "#8A8984"
TEST_COLOR = "black"

MODELS = [
    ("LSTM", "test_predicted_meanseed_lstm.csv", "seeds_mean_prediction", "purple", "-", "(a) LSTM", "test_pred_lstm_seedsmean.pdf"),
    ("GRU", "test_predicted_meanseed_gru.csv", "seeds_mean_prediction", "orange", "-", "(b) GRU", "test_pred_gru_seedsmean.pdf"),
    ("SVR", "test_predicted_svr.csv", "predicted_arrivals", "red", "-", "(c) SVR", "test_pred_svr.pdf"),
    ("GAN (BCE)", "test_predicted_meanseed_gan.csv", "seeds_mean_prediction", "brown", "-", "(d) GAN (BCE)", "test_pred_gan_seedsmean.pdf"),
    ("ARIMAX", "test_predicted_arimax.csv", "predicted_arrivals", "green", "-", "(e) ARIMAX", "test_pred_arima.pdf"),
    ("SARIMAX", "test_predicted_sarimax.csv", "predicted_arrivals", "teal", "-", "(f) SARIMAX", "test_pred_sarimax.pdf"),
    ("ANN", "test_predicted_meanseed_ann.csv", "seeds_mean_prediction", "blue", "-", "(g) ANN", "test_pred_ann_seedsmean.pdf"),
    ("Seasonal naive", "test_predicted_snaive.csv", "predicted_arrivals", "gray", "--", "(h) Seasonal naive", "test_pred_snaive.pdf"),
]

# ===================== DATA ================================
actual = (pd.read_csv("Data/main_dataset.csv", parse_dates=["date"])
          .set_index("date").sort_index()["volume"])

# Each segment includes the first point of the next one, so the line stays continuous
train_seg = actual[actual.index <= VAL_START]
val_seg = actual[(actual.index >= VAL_START) & (actual.index <= TEST_START)]
test_seg = actual[actual.index >= TEST_START]

# Test-set predictions of every model (first column = date)
preds = {label: pd.read_csv(f"Results/{file}", index_col=0, parse_dates=True)[col]
         for label, file, col, *_ in MODELS}


# ===================== FIG. 6: ALL MODELS ON ONE PLOT ======
plt.figure(figsize=(16, 8))

plt.plot(train_seg, color=TRAIN_COLOR, linewidth=2, label="Training set (actual)")
plt.plot(val_seg, color=VAL_COLOR, linewidth=2, label="Validation set (actual)")
plt.plot(test_seg, color=TEST_COLOR, linewidth=2.5, label="Test set (actual)", zorder=10)

# Dashed vertical lines at the split boundaries
plt.axvline(VAL_START, color="gray", linestyle="--", linewidth=1, alpha=0.8)
plt.axvline(TEST_START, color="gray", linestyle="--", linewidth=1, alpha=0.8)

for label, _, _, color, style, _, _ in MODELS:
    plt.plot(preds[label], linewidth=2, color=color, linestyle=style, label=label)

plt.xlabel("Date (month)", fontsize=20)
plt.ylabel("Number of tourists", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=12, frameon=False)
plt.savefig("Results/test_predicted_all.pdf")
plt.show()


# ===================== FIG. 7: ONE PANEL PER MODEL =========
for label, _, _, color, _, title, out_file in MODELS:
    plt.figure(figsize=(16, 8))
    plt.plot(test_seg, linewidth=3, color="black")
    plt.plot(preds[label], linewidth=3, color=color)

    plt.xlabel("Date (month)", fontsize=20)
    plt.ylabel("Number of tourists", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(("Actual", label), loc="upper center", fontsize=18)
    plt.title(title, fontsize=38)

    plt.savefig(f"Results/{out_file}", bbox_inches="tight", pad_inches=0)
    plt.show()