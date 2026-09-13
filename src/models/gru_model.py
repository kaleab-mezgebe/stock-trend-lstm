"""
Gated Recurrent Unit (GRU) Neural Network for Sequence Modeling.
Implements Reset Gate (r_t), Update Gate (z_t), and Candidate Activation (h_tilde).
"""

import math
import random

def sigmoid(x):
    x_clamped = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x_clamped))

def tanh(x):
    x_clamped = max(-20.0, min(20.0, x))
    return math.tanh(x_clamped)

class GRUCell:
    def __init__(self, input_dim=1, hidden_dim=16, seed=42):
        random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        limit = math.sqrt(6.0 / (input_dim + hidden_dim))
        
        # Reset Gate weights (Wr, Ur, br)
        self.Wr = [random.uniform(-limit, limit) for _ in range(hidden_dim)]
        self.Ur = [[random.uniform(-limit, limit) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.br = [0.0 for _ in range(hidden_dim)]
        
        # Update Gate weights (Wz, Uz, bz)
        self.Wz = [random.uniform(-limit, limit) for _ in range(hidden_dim)]
        self.Uz = [[random.uniform(-limit, limit) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.bz = [0.0 for _ in range(hidden_dim)]
        
        # Candidate hidden state weights (Wh, Uh, bh)
        self.Wh = [random.uniform(-limit, limit) for _ in range(hidden_dim)]
        self.Uh = [[random.uniform(-limit, limit) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.bh = [0.0 for _ in range(hidden_dim)]
        
        # Output Dense layer (Wy, by)
        self.Wy = [random.uniform(-limit, limit) for _ in range(hidden_dim)]
        self.by = 0.0

    def forward_step(self, x_t, h_prev):
        h_next = [0.0] * self.hidden_dim
        
        for j in range(self.hidden_dim):
            ur_sum = sum(self.Ur[j][k] * h_prev[k] for k in range(self.hidden_dim))
            uz_sum = sum(self.Uz[j][k] * h_prev[k] for k in range(self.hidden_dim))
            
            # Reset & Update gates
            r_t = sigmoid(self.Wr[j] * x_t + ur_sum + self.br[j])
            z_t = sigmoid(self.Wz[j] * x_t + uz_sum + self.bz[j])
            
            # Candidate activation
            uh_sum = sum(self.Uh[j][k] * (r_t * h_prev[k]) for k in range(self.hidden_dim))
            h_tilde = tanh(self.Wh[j] * x_t + uh_sum + self.bh[j])
            
            # State update
            h_next[j] = (1.0 - z_t) * h_prev[j] + (z_t * h_tilde)
            
        return h_next

    def forward_sequence(self, sequence):
        h = [0.0] * self.hidden_dim
        for x_t in sequence:
            h = self.forward_step(x_t, h)
        y_pred = sum(self.Wy[j] * h[j] for j in range(self.hidden_dim)) + self.by
        return y_pred, h

class GRUPredictor:
    def __init__(self, hidden_dim=16, learning_rate=0.005, epochs=15, seed=42):
        self.model = GRUCell(input_dim=1, hidden_dim=hidden_dim, seed=seed)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_trained = False
        
    def fit(self, X_train, y_train):
        for epoch in range(self.epochs):
            for seq, y_true in zip(X_train, y_train):
                y_pred, h = self.model.forward_sequence(seq)
                error = y_pred - y_true
                grad_y = 2.0 * error
                for j in range(self.model.hidden_dim):
                    grad_Wy = grad_y * h[j]
                    grad_Wy_clipped = max(-1.0, min(1.0, grad_Wy))
                    self.model.Wy[j] -= self.learning_rate * grad_Wy_clipped
                self.model.by -= self.learning_rate * max(-1.0, min(1.0, grad_y))
        self.is_trained = True
        return self
        
    def predict(self, X):
        if X and isinstance(X[0], (int, float)):
            y_pred, _ = self.model.forward_sequence(X)
            return y_pred
            
        predictions = []
        for seq in X:
            y_pred, _ = self.model.forward_sequence(seq)
            predictions.append(y_pred)
        return predictions

GRUTimeSeriesModel = GRUPredictor
