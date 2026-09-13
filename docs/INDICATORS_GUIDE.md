# Quantitative Technical Indicators Mathematical Guide

**Project:** AI-Based Stock Market Prediction & Predictive Sequence Analytics  
**Author:** Kaleab Mezgebe

---

## 1. Introduction

Technical indicators transform non-stationary raw price series ($P_t$) into bounded, momentum-preserving feature matrices. This guide documents the mathematical formulations implemented in `src/features/indicators.py`.

---

## 2. Mathematical Formulations

### 1. Simple Moving Average (SMA)
Calculates the arithmetic mean over a sliding rolling window of size $N$:

$$\text{SMA}_N(t) = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i}$$

### 2. Exponential Moving Average (EMA)
Applies exponentially decaying weighting to prioritize recent observations:

$$\text{EMA}_N(t) = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_N(t-1), \quad \text{where } \alpha = \frac{2}{N + 1}$$

### 3. Relative Strength Index (RSI)
Quantifies the velocity and magnitude of price momentum over $N=14$ periods:

$$\begin{aligned}
U_t &= \max(P_t - P_{t-1}, 0) \\
D_t &= \max(P_{t-1} - P_t, 0) \\
\text{RS} &= \frac{\text{EMA}_{14}(U)}{\text{EMA}_{14}(D)} \\
\text{RSI} &= 100 - \frac{100}{1 + \text{RS}}
\end{aligned}$$

- **Overbought Threshold:** $\text{RSI} \ge 70$
- **Oversold Threshold:** $\text{RSI} \le 30$

### 4. Moving Average Convergence Divergence (MACD)
Captures interactions between short-term (12) and long-term (26) momentum trends:

$$\begin{aligned}
\text{MACD}(t) &= \text{EMA}_{12}(P_t) - \text{EMA}_{26}(P_t) \\
\text{Signal}(t) &= \text{EMA}_{9}(\text{MACD}(t)) \\
\text{Histogram}(t) &= \text{MACD}(t) - \text{Signal}(t)
\end{aligned}$$

### 5. Bollinger Bands
Constructs dynamic volatility envelopes using $K=2$ standard deviations:

$$\begin{aligned}
\mu_t &= \text{SMA}_{20}(P_t) \\
\sigma_t &= \sqrt{\frac{1}{20} \sum_{i=0}^{19} (P_{t-i} - \mu_t)^2} \\
\text{Upper}_t &= \mu_t + 2\sigma_t \\
\text{Lower}_t &= \mu_t - 2\sigma_t \\
\%B_t &= \frac{P_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}
\end{aligned}$$

### 6. Average True Range (ATR)
Measures market volatility incorporating gap-openings:

$$\begin{aligned}
\text{TR}_t &= \max\left( \text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}| \right) \\
\text{ATR}_t &= \text{EMA}_{14}(\text{TR}_t)
\end{aligned}$$

### 7. Stochastic Oscillator (%K, %D)
Assesses the closing price relative to the high-low range over $N=14$ periods:

$$\begin{aligned}
\%K_t &= 100 \times \frac{\text{Close}_t - \min_{14}(\text{Low})}{\max_{14}(\text{High}) - \min_{14}(\text{Low})} \\
\%D_t &= \text{SMA}_3(\%K_t)
\end{aligned}$$

### 8. On-Balance Volume (OBV)
Accumulates directional trading volume to identify institutional accumulation:

$$\text{OBV}_t = \begin{cases}
\text{OBV}_{t-1} + \text{Volume}_t & \text{if } \text{Close}_t > \text{Close}_{t-1} \\
\text{OBV}_{t-1} - \text{Volume}_t & \text{if } \text{Close}_t < \text{Close}_{t-1} \\
\text{OBV}_{t-1} & \text{if } \text{Close}_t = \text{Close}_{t-1}
\end{cases}$$
