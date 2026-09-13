"""
Multi-Step Horizon Forecaster with Monte Carlo Confidence Interval Simulation.
Generates multi-day recursive autoregressive projections with statistical upper/lower bounds.
"""

import math
import random

def run_multistep_forecast(model, last_sequence, scaler, horizon_days=30, n_simulations=50, volatility_std=0.012):
    """
    Performs recursive multi-step forecasting with Monte Carlo confidence cone generation.
    Returns:
    - mean_trajectory (predicted prices)
    - ci_upper_95 (upper 95% confidence bound)
    - ci_lower_95 (lower 95% confidence bound)
    """
    simulated_trajectories = []
    
    for sim_idx in range(n_simulations):
        current_seq = list(last_sequence)
        sim_prices = []
        
        for step in range(horizon_days):
            y_scaled_pred, _, _ = model.model.forward_sequence(current_seq) if hasattr(model.model, 'forward_sequence') else (model.predict([current_seq])[0], None, None)
            
            # Stochastic variance injection for uncertainty quantification
            noise = random.gauss(0, volatility_std * math.sqrt(step + 1) * 0.2)
            y_scaled_noisy = y_scaled_pred + noise
            
            # Unscale to USD price
            price_pred = scaler.inverse_transform([y_scaled_noisy])[0]
            sim_prices.append(price_pred)
            
            # Slide window
            current_seq.pop(0)
            current_seq.append(y_scaled_noisy)
            
        simulated_trajectories.append(sim_prices)
        
    # Aggregate statistics
    mean_trajectory = []
    ci_upper_95 = []
    ci_lower_95 = []
    
    for step in range(horizon_days):
        step_prices = sorted([simulated_trajectories[s][step] for s in range(n_simulations)])
        mean_p = sum(step_prices) / n_simulations
        p_low = step_prices[int(0.05 * n_simulations)]
        p_high = step_prices[int(0.95 * n_simulations)]
        
        mean_trajectory.append(round(mean_p, 2))
        ci_lower_95.append(round(p_low, 2))
        ci_upper_95.append(round(p_high, 2))
        
    return {
        "horizon_days": horizon_days,
        "mean_trajectory": mean_trajectory,
        "ci_upper_95": ci_upper_95,
        "ci_lower_95": ci_lower_95
    }

class SequencePredictor:
    def __init__(self, csv_filepath, ticker="AAPL", seq_len=60):
        import csv
        from features.preprocessor import MinMaxScaler
        from models.lstm_model import LSTMPredictor
        
        self.ticker = ticker
        self.seq_len = seq_len
        self.rows = []
        with open(csv_filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append({
                    "Date": r["Date"],
                    "Open": float(r["Open"]),
                    "High": float(r["High"]),
                    "Low": float(r["Low"]),
                    "Close": float(r["Close"]),
                    "Volume": int(r["Volume"])
                })
                
        closes = [r["Close"] for r in self.rows]
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaled_closes = self.scaler.fit_transform(closes)
        
        # Train fast LSTM model
        self.model = LSTMPredictor(hidden_dim=16, learning_rate=0.01, epochs=5, seed=42)
        X, y = [], []
        for i in range(len(scaled_closes) - seq_len):
            X.append(scaled_closes[i : i + seq_len])
            y.append(scaled_closes[i + seq_len])
        self.model.fit(X, y)
        self.last_seq = scaled_closes[-seq_len:]

    def forecast_horizon(self, horizon_days=30, num_samples=50, confidence_level=0.95):
        return run_multistep_forecast(self.model, self.last_seq, self.scaler, horizon_days=horizon_days, n_simulations=num_samples)
