#!/usr/bin/env python3
"""
Unit tests for MinMaxScaler and Sliding Sequence Lookback generator.
"""

import unittest
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.features.preprocessor import MinMaxScaler1D, TimeSeriesSequenceDataset

class TestPreprocessor(unittest.TestCase):
    def test_minmax_scaler(self):
        scaler = MinMaxScaler1D(feature_range=(0.0, 1.0))
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        scaled = scaler.fit_transform(data)
        self.assertAlmostEqual(scaled[0], 0.0)
        self.assertAlmostEqual(scaled[-1], 1.0)
        self.assertAlmostEqual(scaled[2], 0.5)

        # Inverse transform
        inv = scaler.inverse_transform(scaled)
        for orig, recovered in zip(data, inv):
            self.assertAlmostEqual(orig, recovered, places=4)

    def test_sequence_generator(self):
        series = list(range(100))
        seq_len = 20
        X, y = TimeSeriesSequenceDataset.create_sequences(series, seq_length=seq_len)
        self.assertEqual(len(X), 100 - seq_len)
        self.assertEqual(len(y), 100 - seq_len)
        self.assertEqual(len(X[0]), seq_len)
        self.assertEqual(X[0][-1], 19)
        self.assertEqual(y[0], 20)

    def test_train_test_split(self):
        series = list(range(100))
        X, y = TimeSeriesSequenceDataset.create_sequences(series, seq_length=10)
        X_tr, X_te, y_tr, y_te = TimeSeriesSequenceDataset.train_test_split_chronological(X, y, train_ratio=0.8)
        self.assertEqual(len(X_tr), 72)
        self.assertEqual(len(X_te), 18)

if __name__ == '__main__':
    unittest.main()
