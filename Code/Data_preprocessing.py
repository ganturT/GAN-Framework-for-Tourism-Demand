import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

# %% --------------------------------------- Settings -----------------------------------------------------------------
DATA_FILE = "Data/main_dataset.csv"
FEATURES = ["volume", "search", "sin12", "cos12", "covid"]   
TARGET = "volume"                                           

WINDOW_SIZE = 12      
HORIZON = 1           
TEST_SHARE = 0.20     
N_VAL = 12            

# %% --------------------------------------- Load data ----------------------------------------------------------------
dataset = pd.read_csv(DATA_FILE, parse_dates=["date"]).set_index("date").sort_index()
dataset.to_csv("Results/dataset.csv")

features = dataset[FEATURES].to_numpy(dtype=float)      
target = dataset[[TARGET]].to_numpy(dtype=float)        
n_months, N_FEATURES = features.shape

# %% --------------------------------------- Split ----------------------------------

n_samples = n_months - WINDOW_SIZE - HORIZON + 1                   # 94
n_trainval = int(round(n_samples * (1 - TEST_SHARE)))              # 75
n_train = n_trainval - N_VAL                                       # 63

bounds = {"train": (0, n_train),
          "val": (n_train, n_trainval),
          "test": (n_trainval, n_samples)}


target_pos = np.arange(n_samples) + WINDOW_SIZE
last_train_month = target_pos[n_train - 1] + HORIZON - 1
fit_months = slice(0, last_train_month + 1)

# %% --------------------------------------- Min-max scaling to [-1, 1] (fitted on training months only) ---------------
X_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(features[fit_months])
y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(target[fit_months])
features_scaled = X_scaler.transform(features)
target_scaled = y_scaler.transform(target)

print("Scaler fitted on:", dataset.index[0].date(), "->", dataset.index[last_train_month].date())
print("Scaled target range over the full series:", target_scaled.min(), "->", target_scaled.max())  # max > 1 is expected

dump(y_scaler, open("Data/y_scaler.pkl", "wb"))

# %% --------------------------------------- Build all samples at once with index arrays -------------------------------
input_rows = np.arange(n_samples)[:, None] + np.arange(WINDOW_SIZE)[None, :]   
target_rows = target_pos[:, None] + np.arange(HORIZON)[None, :]              

X_all = features_scaled[input_rows]              
y_all = target_scaled[target_rows, 0]            
dates_all = dataset.index[target_pos]          

# %% --------------------------------------- Хадгалах ----------------------------------------------------------
for name, (start, end) in bounds.items():
    np.save(f"Data/X_{name}.npy", X_all[start:end])
    np.save(f"Data/y_{name}.npy", y_all[start:end])
    np.save(f"Data/index_{name}.npy", dates_all[start:end])
    print(f"{name:<5}: {end - start:3d} samples | target months "
          f"{dates_all[start].strftime('%Y-%m')} -> {dates_all[end - 1].strftime('%Y-%m')} | "
          f"X {X_all[start:end].shape}")

assert sum(end - start for start, end in bounds.values()) == n_samples

# %% --------------------------------------- Plot the dataset ---------------------------------------------------------
plt.figure(figsize=(16, 8))
plt.bar(dataset.index, dataset[TARGET], color="blue", width=10)
plt.xlabel("Date(month)", fontsize=20)
plt.ylabel("Number of tourists", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("Results/dataset_plot.pdf")
plt.show()