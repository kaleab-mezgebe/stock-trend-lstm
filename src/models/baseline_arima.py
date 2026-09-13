"""
Baseline AutoRegressive & Moving Average (ARIMA-style) Linear Benchmark.
Fits AR(p) model with lag coefficients for performance comparison against deep recurrent networks.
"""

class ARIMABaseline:
    def __init__(self, p_lags=5, learning_rate=0.01, epochs=25):
        self.p_lags = p_lags
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [1.0 / p_lags for _ in range(p_lags)]
        self.intercept = 0.0
        self.is_trained = False
        
    def fit(self, X_train, y_train):
        for _ in range(self.epochs):
            for seq, y_true in zip(X_train, y_train):
                # Extract last p_lags values
                lags = seq[-self.p_lags:]
                y_pred = sum(w * x for w, x in zip(self.weights, lags)) + self.intercept
                error = y_pred - y_true
                grad = 2.0 * error
                for k in range(self.p_lags):
                    self.weights[k] -= self.learning_rate * grad * lags[k]
                self.intercept -= self.learning_rate * grad
        self.is_trained = True
        return self
        
    def predict(self, X):
        predictions = []
        for seq in X:
            lags = seq[-self.p_lags:]
            y_pred = sum(w * x for w, x in zip(self.weights, lags)) + self.intercept
            predictions.append(y_pred)
        return predictions
