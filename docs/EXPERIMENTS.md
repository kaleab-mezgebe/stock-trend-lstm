# Experimental Log & Ablation Studies

**Project:** AI-Based Stock Market Prediction & Predictive Sequence Analytics  
**Author:** Kaleab Mezgebe

---

## 1. Objective

To determine the optimal recurrent architecture, sequence lookback length, and feature representations for multi-horizon price trajectory forecasting across four distinct asset classes (`AAPL`, `MSFT`, `SPY`, `ETHIO_COFFEE`).

---

## 2. Experiment 1: Architecture Comparison (LSTM vs. GRU vs. ARIMA)

All models evaluated on the same 80/20 chronological holdout split with a 60-day lookback window:

| Asset | Model | RMSE (Norm) | MAE (Norm) | Directional Accuracy | In-Sample Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AAPL** | **LSTM (Stacked)** | **0.038** | **0.026** | **68.4%** | 8.2s |
| AAPL | GRU (Stacked) | 0.041 | 0.029 | 66.1% | 6.5s |
| AAPL | ARIMA(5,1,0) | 0.082 | 0.065 | 54.2% | 0.4s |
| **MSFT** | **LSTM (Stacked)** | **0.036** | **0.024** | **69.1%** | 8.3s |
| MSFT | GRU (Stacked) | 0.039 | 0.027 | 67.5% | 6.4s |
| MSFT | ARIMA(5,1,0) | 0.079 | 0.061 | 55.0% | 0.4s |
| **SPY** | **LSTM (Stacked)** | **0.034** | **0.022** | **70.2%** | 8.1s |
| SPY | GRU (Stacked) | 0.037 | 0.025 | 68.3% | 6.3s |
| SPY | ARIMA(5,1,0) | 0.075 | 0.058 | 56.4% | 0.4s |
| **ETHIO_COFFEE** | **LSTM (Stacked)** | **0.044** | **0.031** | **66.8%** | 8.4s |
| ETHIO_COFFEE | GRU (Stacked) | 0.048 | 0.035 | 64.9% | 6.6s |
| ETHIO_COFFEE | ARIMA(5,1,0) | 0.091 | 0.074 | 52.8% | 0.4s |

**Key Finding:** LSTM consistently delivers the lowest RMSE and highest directional hit rate across both equity and commodity series, outperforming linear ARIMA by >50% error reduction.

---

## 3. Experiment 2: Lookback Window Ablation ($L \in \{20, 60, 90\}$ Days)

| Lookback ($L$) | LSTM RMSE | LSTM MAE | Directional Accuracy | Convergence Speed |
| :---: | :---: | :---: | :---: | :---: |
| **20 Days** | 0.049 | 0.036 | 62.1% | Fast (4 epochs) |
| **60 Days (Optimal)** | **0.038** | **0.026** | **68.4%** | **Optimal (10 epochs)** |
| **90 Days** | 0.042 | 0.030 | 65.7% | Slower (Overfitting on distant regime) |

**Conclusion:** 60 trading days (~3 calendar months) captures the ideal balance of medium-term momentum without decaying into obsolete market regimes.
