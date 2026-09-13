#!/usr/bin/env python3
"""
Generate multi-year daily financial OHLCV time-series data for stock-trend-lstm.
Tickers:
1. AAPL (Apple Inc. - Tech Equity Benchmark)
2. MSFT (Microsoft Corp. - Enterprise SaaS / AI Benchmark)
3. SPY (S&P 500 ETF - Macro Equity Benchmark)
4. ETHIO_COFFEE (Ethiopian Specialty Arabica Coffee Commodity Index - Agricultural Benchmark)
"""

import os
import csv
import math
import random
import datetime

def generate_stock_series(ticker, start_date_str, n_days, base_price, drift, volatility, seed):
    random.seed(seed)
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    
    records = []
    current_price = base_price
    
    # 12-month annual seasonality + cyclical momentum
    for i in range(n_days):
        date_curr = start_date + datetime.timedelta(days=i)
        if date_curr.weekday() >= 5: # Skip weekends
            continue
            
        day_of_year = date_curr.timetuple().tm_yday
        seasonal_shock = 0.0015 * math.sin(2 * math.pi * day_of_year / 365.25)
        
        # Jump diffusion / realistic returns
        shock = random.gauss(drift + seasonal_shock, volatility)
        open_price = current_price * (1 + random.gauss(0, volatility * 0.3))
        close_price = current_price * (1 + shock)
        
        # High and Low
        high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, volatility * 0.6)))
        low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, volatility * 0.6)))
        
        # Volume
        base_vol = 45000000 if ticker in ["AAPL", "SPY"] else (25000000 if ticker == "MSFT" else 150000)
        vol_multiplier = (abs(close_price - open_price) / open_price) * 20.0 + random.uniform(0.7, 1.4)
        volume = int(base_vol * vol_multiplier)
        
        records.append({
            "Date": date_curr.strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Open": round(open_price, 2),
            "High": round(high_price, 2),
            "Low": round(low_price, 2),
            "Close": round(close_price, 2),
            "AdjClose": round(close_price, 2),
            "Volume": volume
        })
        current_price = close_price
        
    return records

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    datasets = [
        ("AAPL", "2021-01-01", 1100, 130.0, 0.00065, 0.016, 42),
        ("MSFT", "2021-01-01", 1100, 220.0, 0.00075, 0.015, 101),
        ("SPY", "2021-01-01", 1100, 375.0, 0.00045, 0.011, 202),
        ("ETHIO_COFFEE", "2021-01-01", 1100, 185.0, 0.00055, 0.019, 303)
    ]
    
    for ticker, start_date, n_days, base_price, drift, vol, seed in datasets:
        records = generate_stock_series(ticker, start_date, n_days, base_price, drift, vol, seed)
        csv_path = os.path.join(raw_dir, f"{ticker.lower()}_historical_daily.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Generated {len(records)} daily records for {ticker} -> {csv_path}")

if __name__ == "__main__":
    main()
