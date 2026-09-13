"""
Data Preprocessor for Time-Series Sequence Modeling (LSTM / GRU).
Implements sliding lookback window generator and MinMax feature scaling.
"""

class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None
        
    def fit(self, data):
        self.min_val = min(data)
        self.max_val = max(data)
        return self
        
    def transform(self, data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        span = self.max_val - self.min_val
        if span == 0:
            return [0.5 for _ in data]
        min_out, max_out = self.feature_range
        return [((x - self.min_val) / span) * (max_out - min_out) + min_out for x in data]
        
    def fit_transform(self, data):
        return self.fit(data).transform(data)
        
    def inverse_transform(self, scaled_data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        span = self.max_val - self.min_val
        min_out, max_out = self.feature_range
        return [((y - min_out) / (max_out - min_out)) * span + self.min_val for y in scaled_data]

def create_sequences(data_series, seq_length=60):
    """
    Transforms a 1D time-series into (X, y) sliding window sequences.
    X: list of lists of length seq_length
    y: list of target next-step values
    """
    X, y = [], []
    for i in range(len(data_series) - seq_length):
        X.append(data_series[i : i + seq_length])
        y.append(data_series[i + seq_length])
    return X, y

def train_test_split_temporal(X, y, train_ratio=0.8):
    """
    Chronological train-test split (avoids lookahead data leakage).
    """
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

# Compatibility aliases
MinMaxScaler1D = MinMaxScaler

class TimeSeriesSequenceDataset:
    create_sequences = staticmethod(create_sequences)
    train_test_split_chronological = staticmethod(train_test_split_temporal)
