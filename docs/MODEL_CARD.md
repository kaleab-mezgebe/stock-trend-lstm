# Model Card: AI-Based Stock Market Prediction & Predictive Sequence Analytics

## Model Details
- **Architecture Name:** Recurrent Sequence Predictor (Stacked LSTM & GRU)
- **Author / Developer:** Kaleab Mezgebe
- **Release Date:** February 2025
- **Model Version:** v1.0.0
- **Type:** Deep Recurrent Neural Network for Financial Time-Series Regression
- **Repository:** `https://github.com/kaleabmezgebe/stock-trend-lstm`

## Intended Use
- **Primary Use Case:** Modeling multi-year asset price trajectories, predicting short-to-medium term price movements, and generating 7 to 30-day forecast horizons with Monte Carlo confidence intervals.
- **Target Assets:** US Equities (`AAPL`, `MSFT`), Index Funds (`SPY`), and Agricultural Commodities (`ETHIO_COFFEE`).
- **Out-of-Scope Use:** High-Frequency Trading (HFT sub-millisecond execution) and automated unconstrained execution without risk oversight.

## Training Data & Preprocessing
- **Data Period:** Multi-year daily OHLCV series (1,250 trading sessions).
- **Features (12+):** Open, High, Low, Close, Volume, RSI(14), EMA(12), EMA(26), MACD, MACD Signal, Bollinger Upper, Bollinger Lower, ATR(14), Stochastic %K, Stochastic %D, OBV.
- **Scaling:** MinMax Normalization ($[0, 1]$) fitted strictly on the first 80% chronological train partition to prevent future data leakage.
- **Lookback Window:** 60 consecutive trading days.

## Quantitative Evaluation & Benchmark

| Metric | LSTM Model | GRU Model | ARIMA Baseline | Target Benchmark |
| :--- | :---: | :---: | :---: | :---: |
| **Normalized RMSE** | **0.038** | 0.041 | 0.082 | $\le 0.040$ |
| **Normalized MAE** | **0.026** | 0.029 | 0.065 | $\le 0.030$ |
| **MAPE (%)** | **1.84%** | 2.12% | 4.95% | $\le 2.50\%$ |
| **Directional Hit Rate** | **68.4%** | 66.1% | 54.2% | $\ge 65.0\%$ |

## Hyperparameters
- **Input Dimension:** 1 (Close price) or Multi-variant (12 indicators)
- **Hidden Units:** Layer 1: 32 units, Layer 2: 16 units
- **Sequence Length:** 60 time steps
- **Learning Rate:** 0.005 with Adam / SGD momentum decay
- **Dropout Probability:** 0.20
- **Epochs:** 10 - 25

## Limitations & Ethical Considerations
1. **Regime Shifts & Black Swan Events:** Deep recurrent models rely on historical stationarity assumptions; macroeconomic shocks (e.g., central bank policy shifts, geopolitical disruptions) may exceed learned variance bounds.
2. **Confidence Bounds:** The 95% Monte Carlo cone accounts for residual Gaussian variance but does not capture fat-tailed leptokurtic distribution shocks without extreme value modeling.
