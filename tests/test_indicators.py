#!/usr/bin/env python3
"""
Unit tests for Technical Indicator mathematical calculations.
"""

import unittest
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.features.indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        self.closes = [100.0 + i * 0.5 + (1.0 if i % 2 == 0 else -1.0) for i in range(60)]
        self.highs = [c + 1.5 for c in self.closes]
        self.lows = [c - 1.5 for c in self.closes]
        self.volumes = [1000000 + i * 5000 for i in range(60)]
        self.records = [
            {
                "Date": f"2024-01-{i+1:02d}",
                "Open": self.closes[i] - 0.2,
                "High": self.highs[i],
                "Low": self.lows[i],
                "Close": self.closes[i],
                "Volume": self.volumes[i]
            }
            for i in range(60)
        ]

    def test_sma(self):
        sma20 = TechnicalIndicators.sma(self.closes, 20)
        self.assertEqual(len(sma20), len(self.closes))
        self.assertAlmostEqual(sma20[19], sum(self.closes[:20]) / 20.0, places=4)

    def test_ema(self):
        ema12 = TechnicalIndicators.ema(self.closes, 12)
        self.assertEqual(len(ema12), len(self.closes))
        self.assertIsNotNone(ema12[-1])
        self.assertGreater(ema12[-1], 0)

    def test_rsi_bounds(self):
        rsi = TechnicalIndicators.rsi(self.closes, 14)
        valid_rsi = [r for r in rsi if r is not None]
        self.assertTrue(len(valid_rsi) > 0)
        for r in valid_rsi:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 100.0)

    def test_bollinger_bands(self):
        upper, middle, lower, width = TechnicalIndicators.bollinger_bands(self.closes, 20, 2.0)
        self.assertEqual(len(upper), len(self.closes))
        for u, l in zip(upper, lower):
            if u is not None and l is not None:
                self.assertGreaterEqual(u, l)

    def test_atr(self):
        atr = TechnicalIndicators.atr(self.highs, self.lows, self.closes, 14)
        valid_atr = [a for a in atr if a is not None]
        self.assertTrue(len(valid_atr) > 0)
        for a in valid_atr:
            self.assertGreater(a, 0)

    def test_stochastic(self):
        k, d = TechnicalIndicators.stochastic_oscillator(self.highs, self.lows, self.closes, 14, 3)
        valid_k = [val for val in k if val is not None]
        self.assertTrue(len(valid_k) > 0)
        for val in valid_k:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 100.0)

    def test_compute_all(self):
        res = TechnicalIndicators.compute_all(self.records)
        self.assertEqual(len(res), len(self.records))
        expected_keys = ["SMA_20", "EMA_12", "EMA_26", "RSI_14", "MACD", "MACD_Signal",
                         "BB_Upper", "BB_Lower", "BB_Width", "ATR_14", "Stoch_K", "Stoch_D", "OBV"]
        for k in expected_keys:
            self.assertIn(k, res[-1])

if __name__ == '__main__':
    unittest.main()
