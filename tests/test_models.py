#!/usr/bin/env python3
"""
Unit tests for LSTM, GRU, ARIMA baselines, and evaluation metrics.
"""

import unittest
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.lstm_model import LSTMTimeSeriesModel
from src.models.gru_model import GRUTimeSeriesModel
from src.models.baseline_arima import ARIMABaseline
from src.models.evaluator import TimeSeriesEvaluator

class TestModelsAndMetrics(unittest.TestCase):
    def test_metrics(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 1.9, 3.2, 3.9, 5.1]
        metrics = TimeSeriesEvaluator.evaluate_all(y_true, y_pred)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertIn("Directional_Accuracy", metrics)
        self.assertLess(metrics["RMSE"], 0.2)
        self.assertGreater(metrics["Directional_Accuracy"], 50.0)

    def test_lstm_forward(self):
        model = LSTMTimeSeriesModel(hidden_dim=8)
        seq = [0.1 * i for i in range(10)]
        pred = model.predict(seq)
        self.assertIsInstance(pred, float)

    def test_gru_forward(self):
        model = GRUTimeSeriesModel(hidden_dim=8)
        seq = [0.1 * i for i in range(10)]
        pred = model.predict(seq)
        self.assertIsInstance(pred, float)

    def test_arima_fit_predict(self):
        train_X = [[0.1 * (i + j) for j in range(10)] for i in range(20)]
        train_y = [0.1 * (i + 10) for i in range(20)]
        test_X = [[0.1 * (20 + j) for j in range(10)]]
        
        arima = ARIMABaseline(p_lags=5)
        arima.fit(train_X, train_y)
        preds = arima.predict(test_X)
        self.assertEqual(len(preds), 1)
        self.assertIsInstance(preds[0], float)

if __name__ == '__main__':
    unittest.main()
