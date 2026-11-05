# A Generative Adversarial Network Framework for Tourism Demand Forecasting: A Comparison of Loss Functions

This repository contains the code and data for the study "A Generative Adversarial Network Framework for Tourism Demand Forecasting: A Comparison of Loss Functions".

The study forecasts monthly inbound tourist arrivals to Mongolia (January 2017 – October 2025) with a GAN that uses a GRU-based generator and a Conv1D-based discriminator, and compares three loss functions on the same architecture: binary cross-entropy (BCE), BCE combined with mean squared error (BCE + MSE), and the Wasserstein loss with gradient penalty (WGAN-GP). The GAN variants are compared with a seasonal naive forecast and with ARIMAX, SARIMAX, SVR, ANN, LSTM and GRU benchmarks.

## Repository structure

```
tourism-gan-forecasting/
├── Code/                       Python scripts (see the table below)
├── Data/
│   └── main_dataset.csv        Monthly dataset used in the study
├── Results/                    Test predictions, metrics, selected hyperparameters and tuning logs
├── requirements.txt
├── LICENSE
└── README.md
```

`Code/Data_preprocessing.py` additionally writes the model inputs to `Data/` (`X_*.npy`, `y_*.npy`, `index_*.npy` and `y_scaler.pkl`).

## Data

`Data/main_dataset.csv` contains 106 monthly observations (January 2017 – October 2025) with the following columns:

| Column   | Description |
|----------|-------------|
| `date`   | First day of the month (YYYY-MM-DD) |
| `volume` | Number of inbound tourist arrivals ([National Statistics Office of Mongolia](https://www.1212.mn)) |
| `search` | log(1 + mean of five Google Trends search indices) |
| `sin12`, `cos12` | Cyclical month terms, sin(2πm/12) and cos(2πm/12) |
| `covid`  | COVID-19 dummy (1 from March 2020 to March 2022, 0 otherwise) |

The five Google Trends search terms are "Mongolia tours", "flight to Ulaanbaatar", "Mongolia visa", "visit Mongolia" and "best time to visit Mongolia" (worldwide, all categories, retrieved on 10 October 2026). Google Trends data are sampled, so indices downloaded on a different day may differ slightly from those used in the paper.

The data are split chronologically into training (63 samples, Jan 2018 – Mar 2023), validation (12 samples, Apr 2023 – Mar 2024) and test (19 samples, Apr 2024 – Oct 2025) sets, using a 12-month input window. The min–max scalers are fitted on the training period only.

## Installation

Python 3.12.2 was used. Install the required packages with:

```bash
pip install -r requirements.txt
```

Main package versions: TensorFlow 2.18.0, Keras 3.8.0, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1, statsmodels 0.14.4. No GPU is required; all models were trained on a CPU (Intel Core i7-1165G7, 8 GB RAM, Windows 11).

## How to run

Run all scripts from the **root folder** of the repository (the scripts use the relative paths `Data/` and `Results/`). The GAN scripts run each training run in a separate process (`multiprocessing`, spawn), so run them as scripts from the command line rather than inside a Jupyter notebook.

```bash
# 1. Prepare the windows, the chronological split and the min-max scalers
python Code/Data_preprocessing.py

# 2. Train and evaluate the three GAN variants
python Code/GAN_bce.py
python Code/GAN_bce_mse.py
python Code/GAN_wgan_gp.py

# 3. Train and evaluate the benchmarks
python Code/BM_ANN.py
python Code/BM_LSTM.py
python Code/BM_GRU.py
python Code/BM_SVR.py
python Code/BM_ARIMAX.py
python Code/BM_SARIMAX.py
python Code/BM_SNAIVE.py

# 4. Build Table 2, the Diebold-Mariano test (Table 3) and the prediction figures
python Code/PR_testset_table.py
python Code/PR_dm_test.py
python Code/PR_all_models_plots.py
```

| Script | Output | Paper |
|--------|--------|-------|
| `Data_preprocessing.py` | Training/validation/test arrays, scalers, dataset plot | Data description, Fig. 3 |
| `GAN_bce.py` | GAN with the BCE loss: grid search, 4-seed training, test predictions, D/G loss plot | Tables 1–2, Fig. 7a |
| `GAN_bce_mse.py` | GAN with the BCE + MSE loss (same steps) | Tables 1–2, Fig. 7b |
| `GAN_wgan_gp.py` | GAN with the WGAN-GP loss (same steps) | Tables 1–2, Fig. 7c |
| `BM_ANN.py`, `BM_LSTM.py`, `BM_GRU.py` | Neural benchmarks: grid search, 4-seed training, loss curves | Tables 1–2, Fig. 6 |
| `BM_SVR.py`, `BM_ARIMAX.py`, `BM_SARIMAX.py` | Machine learning and statistical benchmarks | Tables 1–2 |
| `BM_SNAIVE.py` | Seasonal naive forecast | Table 2 |
| `PR_testset_table.py` | Performance comparison table (CSV and LaTeX) | Table 2 |
| `PR_dm_test.py` | Diebold–Mariano test between the GAN loss functions (CSV and LaTeX) | Table 3 |
| `PR_all_models_plots.py` | Test-set prediction plots | Figs. 4–5 |

**Hyperparameter search.** In the neural scripts, the full search grids of Table 1 are listed first and are then overridden by the selected values (e.g. `LR_GRID = [0.0002]`), so that the final results can be reproduced quickly. To repeat the full grid search, delete the lines with the selected values. The full GAN search is slow on a CPU: 7 learning rates × 5 generator sizes × 5 discriminator sizes = 175 configurations × 3 seeds for the BCE and WGAN-GP losses, and four times as many for BCE + MSE (λ ∈ {0.1, 1, 10, 100}). Every tuning run is saved immediately to `Results/tuning_runs_<model>.csv`; if a search is interrupted, running the script again skips the runs already saved.

The neural models are tuned with three seeds (10, 11, 12), and the configuration with the lowest mean + standard deviation of the validation RMSE is selected. The final models are trained with four seeds (0, 1, 2, 3), and results are reported as mean ± standard deviation over these seeds. The test set is not used for any model selection. Because of differences in hardware and library versions, re-running the neural models may give slightly different numbers from those in the paper; the predictions and metrics used in the paper are stored in `Results/`.



## GAN model design

The GRU generator and Conv1D discriminator architecture of the GAN model is adapted from Lin et al. (2021):

> H. Lin, C. Chen, G. Huang and A. Jafari, "Stock price prediction using generative adversarial networks," *Journal of Computer Science*, vol. 17, pp. 188–196, 2021. https://doi.org/10.3844/jcssp.2021.188.196

## License

This project is released under the [MIT License](LICENSE).