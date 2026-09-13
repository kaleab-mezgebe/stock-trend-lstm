"""
Technical Indicators Engine for Financial Time-Series Analytics.
Implements mathematical formulas for 12+ quantitative technical indicators.
Pure Python standard library implementation for zero-dependency portability.
"""

import math

def calculate_sma(prices, period):
    smas = [None] * len(prices)
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1 : i + 1]
        smas[i] = sum(window) / period
    return smas

def calculate_ema(prices, period):
    emas = [None] * len(prices)
    if len(prices) < period:
        return emas
    
    # Initialize with SMA of first period
    initial_sma = sum(prices[:period]) / period
    emas[period - 1] = initial_sma
    multiplier = 2.0 / (period + 1.0)
    
    for i in range(period, len(prices)):
        emas[i] = (prices[i] - emas[i - 1]) * multiplier + emas[i - 1]
    return emas

def calculate_rsi(prices, period=14):
    rsi = [None] * len(prices)
    if len(prices) <= period:
        return rsi
    
    gains = []
    losses = []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
        
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
        
    for i in range(period + 1, len(prices)):
        curr_gain = gains[i - 1]
        curr_loss = losses[i - 1]
        
        avg_gain = (avg_gain * (period - 1) + curr_gain) / period
        avg_loss = (avg_loss * (period - 1) + curr_loss) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    
    macd_line = [None] * len(prices)
    for i in range(len(prices)):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]
            
    # Calculate Signal line as EMA of MACD Line
    valid_indices = [i for i, val in enumerate(macd_line) if val is not None]
    valid_macd = [macd_line[i] for i in valid_indices]
    
    signal_sub = calculate_ema(valid_macd, signal_period)
    signal_line = [None] * len(prices)
    histogram = [None] * len(prices)
    
    for idx_sub, orig_idx in enumerate(valid_indices):
        if signal_sub[idx_sub] is not None:
            signal_line[orig_idx] = signal_sub[idx_sub]
            histogram[orig_idx] = macd_line[orig_idx] - signal_line[orig_idx]
            
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, num_std=2.0):
    upper_band = [None] * len(prices)
    lower_band = [None] * len(prices)
    middle_band = [None] * len(prices)
    band_width = [None] * len(prices)
    
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1 : i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std_dev = math.sqrt(variance)
        
        middle_band[i] = mean
        upper_band[i] = mean + (num_std * std_dev)
        lower_band[i] = mean - (num_std * std_dev)
        band_width[i] = (upper_band[i] - lower_band[i]) / mean if mean != 0 else 0.0
        
    return upper_band, middle_band, lower_band, band_width

def calculate_atr(highs, lows, closes, period=14):
    true_ranges = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(tr1, tr2, tr3))
        
    atr = [None] * len(closes)
    if len(true_ranges) < period:
        return atr
        
    atr[period - 1] = sum(true_ranges[:period]) / period
    for i in range(period, len(true_ranges)):
        atr[i] = (atr[i - 1] * (period - 1) + true_ranges[i]) / period
    return atr

def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
    stoch_k = [None] * len(closes)
    for i in range(k_period - 1, len(closes)):
        window_high = max(highs[i - k_period + 1 : i + 1])
        window_low = min(lows[i - k_period + 1 : i + 1])
        denom = window_high - window_low
        if denom == 0:
            stoch_k[i] = 50.0
        else:
            stoch_k[i] = ((closes[i] - window_low) / denom) * 100.0
            
    # %D is SMA of %K
    stoch_d = [None] * len(closes)
    for i in range(len(closes)):
        valid_k = [stoch_k[j] for j in range(max(0, i - d_period + 1), i + 1) if stoch_k[j] is not None]
        if len(valid_k) == d_period:
            stoch_d[i] = sum(valid_k) / d_period
            
    return stoch_k, stoch_d

def calculate_obv(closes, volumes):
    obv = [0] * len(closes)
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv

def compute_all_indicators(records):
    """
    Computes all 12+ technical indicators for a list of OHLCV records.
    Returns enriched records.
    """
    closes = [r["Close"] for r in records]
    opens = [r["Open"] for r in records]
    highs = [r["High"] for r in records]
    lows = [r["Low"] for r in records]
    volumes = [r["Volume"] for r in records]
    
    sma_20 = calculate_sma(closes, 20)
    sma_50 = calculate_sma(closes, 50)
    sma_200 = calculate_sma(closes, 200)
    ema_12 = calculate_ema(closes, 12)
    ema_26 = calculate_ema(closes, 26)
    macd, signal, hist = calculate_macd(closes)
    rsi_14 = calculate_rsi(closes, 14)
    bb_upper, bb_mid, bb_lower, bb_width = calculate_bollinger_bands(closes, 20)
    atr_14 = calculate_atr(highs, lows, closes, 14)
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes, 14, 3)
    obv = calculate_obv(closes, volumes)
    
    enriched = []
    for i, r in enumerate(records):
        item = dict(r)
        item["SMA_20"] = round(sma_20[i], 2) if sma_20[i] is not None else None
        item["SMA_50"] = round(sma_50[i], 2) if sma_50[i] is not None else None
        item["SMA_200"] = round(sma_200[i], 2) if sma_200[i] is not None else None
        item["EMA_12"] = round(ema_12[i], 2) if ema_12[i] is not None else None
        item["EMA_26"] = round(ema_26[i], 2) if ema_26[i] is not None else None
        item["MACD"] = round(macd[i], 3) if macd[i] is not None else None
        item["MACD_Signal"] = round(signal[i], 3) if signal[i] is not None else None
        item["MACD_Hist"] = round(hist[i], 3) if hist[i] is not None else None
        item["RSI_14"] = round(rsi_14[i], 2) if rsi_14[i] is not None else None
        item["BB_Upper"] = round(bb_upper[i], 2) if bb_upper[i] is not None else None
        item["BB_Middle"] = round(bb_mid[i], 2) if bb_mid[i] is not None else None
        item["BB_Lower"] = round(bb_lower[i], 2) if bb_lower[i] is not None else None
        item["BB_Width"] = round(bb_width[i], 4) if bb_width[i] is not None else None
        item["ATR_14"] = round(atr_14[i], 2) if atr_14[i] is not None else None
        item["Stoch_K"] = round(stoch_k[i], 2) if stoch_k[i] is not None else None
        item["Stoch_D"] = round(stoch_d[i], 2) if stoch_d[i] is not None else None
        item["OBV"] = obv[i]
        
        # Daily Return
        if i > 0 and closes[i-1] != 0:
            ret = (closes[i] - closes[i-1]) / closes[i-1]
            item["Daily_Return"] = round(ret, 4)
        else:
            item["Daily_Return"] = 0.0
            
        enriched.append(item)
        
    return enriched

class TechnicalIndicators:
    """Class wrapper for technical indicator computations."""
    sma = staticmethod(calculate_sma)
    ema = staticmethod(calculate_ema)
    rsi = staticmethod(calculate_rsi)
    macd = staticmethod(calculate_macd)
    bollinger_bands = staticmethod(calculate_bollinger_bands)
    atr = staticmethod(calculate_atr)
    stochastic_oscillator = staticmethod(calculate_stochastic)
    obv = staticmethod(calculate_obv)
    compute_all = staticmethod(compute_all_indicators)
