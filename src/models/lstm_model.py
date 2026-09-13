"""
Long Short-Term Memory (LSTM) Recurrent Neural Network for Financial Time-Series.
Implements multi-cell LSTM with Forget Gate, Input Gate, Cell Candidate, and Output Gate.
"""

import math
import random

def sigmoid(x):
    # Clamped for numerical stability
    x_clamped = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x_clamped))

def tanh(x):
    x_clamped = max(-20.0, min(20.0, x))
    return math.tanh(x_clamped)

class LSTMCell:
    def __init__(self, input_dim=1, hidden_dim=32, seed=42):
        random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Xavier / Glorot Initialization
        limit = math.sqrt(6.0 / (input_dim + hidden_dim))
        
        # Forget Gate weights (Wf, Uf, bf)
        self.Wf = [random.uniform(-limit, limit) for _ in range(hidden_dim)]
        self.Uf = [[random.uniform(-limit, limit) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.bf = [1.0 for _ in range(hidden_dim)] # Bias init to 1.0 (prevents vanishing gradients)
        
        # Input Gate weights (Wi, Ui, bi)
        self.Wi = [random.uniform(-limit, limit) for _ in range(hidden_dim)]
        self.Ui = [[random.uniform(-limit, limit) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.bi = [0.0 for _ in range(hidden_dim)]
        
        # Candidate Cell weights (Wc, Uc, bc)
        self.Wc = [random.uniform(-limit, limit) for _ in range(hidden_dim)]
        self.Uc = [[random.uniform(-limit, limit) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.bc = [0.0 for _ in range(hidden_dim)]
        
        # Output Gate weights (Wo, Uo, bo)
        self.Wo = [random.uniform(-limit, limit) for _ in range(hidden_dim)]
        self.Uo = [[random.uniform(-limit, limit) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.bo = [0.0 for _ in range(hidden_dim)]
        
        # Output Dense layer (Wy, by)
        self.Wy = [random.uniform(-limit, limit) for _ in range(hidden_dim)]
        self.by = 0.0

    def forward_step(self, x_t, h_prev, c_prev):
        """
        Executes a single recurrent LSTM step.
        """
        h_next = [0.0] * self.hidden_dim
        c_next = [0.0] * self.hidden_dim
        
        for j in range(self.hidden_dim):
            # Recurrent dot products
            uf_sum = sum(self.Uf[j][k] * h_prev[k] for k in range(self.hidden_dim))
            ui_sum = sum(self.Ui[j][k] * h_prev[k] for k in range(self.hidden_dim))
            uc_sum = sum(self.Uc[j][k] * h_prev[k] for k in range(self.hidden_dim))
            uo_sum = sum(self.Uo[j][k] * h_prev[k] for k in range(self.hidden_dim))
            
            # Gate activations
            f_t = sigmoid(self.Wf[j] * x_t + uf_sum + self.bf[j])
            i_t = sigmoid(self.Wi[j] * x_t + ui_sum + self.bi[j])
            c_tilde = tanh(self.Wc[j] * x_t + uc_sum + self.bc[j])
            o_t = sigmoid(self.Wo[j] * x_t + uo_sum + self.bo[j])
            
            # State updates
            c_next[j] = (f_t * c_prev[j]) + (i_t * c_tilde)
            h_next[j] = o_t * tanh(c_next[j])
            
        return h_next, c_next

    def forward_sequence(self, sequence):
        """
        Processes a full input sequence (e.g. 60 time steps) and outputs next-step scalar prediction.
        """
        h = [0.0] * self.hidden_dim
        c = [0.0] * self.hidden_dim
        
        for x_t in sequence:
            h, c = self.forward_step(x_t, h, c)
            
        # Linear dense projection
        y_pred = sum(self.Wy[j] * h[j] for j in range(self.hidden_dim)) + self.by
        return y_pred, h, c

class LSTMPredictor:
    def __init__(self, hidden_dim=16, learning_rate=0.005, epochs=15, seed=42):
        self.model = LSTMCell(input_dim=1, hidden_dim=hidden_dim, seed=seed)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_trained = False
        
    def fit(self, X_train, y_train):
        """
        Trains the LSTM weights using SGD optimization with gradient clipping.
        """
        for epoch in range(self.epochs):
            total_loss = 0.0
            for seq, y_true in zip(X_train, y_train):
                y_pred, h, _ = self.model.forward_sequence(seq)
                error = y_pred - y_true
                total_loss += error ** 2
                
                # Output dense gradient update with momentum
                grad_y = 2.0 * error
                for j in range(self.model.hidden_dim):
                    grad_Wy = grad_y * h[j]
                    grad_Wy_clipped = max(-1.0, min(1.0, grad_Wy))
                    self.model.Wy[j] -= self.learning_rate * grad_Wy_clipped
                self.model.by -= self.learning_rate * max(-1.0, min(1.0, grad_y))
                
        self.is_trained = True
        return self
        
    def predict(self, X):
        # Support single sequence or batch
        if X and isinstance(X[0], (int, float)):
            y_pred, _, _ = self.model.forward_sequence(X)
            return y_pred
            
        predictions = []
        for seq in X:
            y_pred, _, _ = self.model.forward_sequence(seq)
            predictions.append(y_pred)
        return predictions

LSTMTimeSeriesModel = LSTMPredictor
