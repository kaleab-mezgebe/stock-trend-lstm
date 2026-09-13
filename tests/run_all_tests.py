#!/usr/bin/env python3
"""
Master Test Suite Runner for AI-Based Stock Market Prediction & Predictive Sequence Analytics.
Author: Kaleab Mezgebe (2025)
"""

import unittest
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

def run_suite():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=SCRIPT_DIR, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    print("=" * 70)
    print(" RUNNING MASTER TEST SUITE: stock-trend-lstm")
    print("=" * 70)
    result = runner.run(suite)
    print("=" * 70)
    if result.wasSuccessful():
        print(" [✓] ALL UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f" [✗] TEST FAILURES DETECTED: {len(result.failures)} Failures, {len(result.errors)} Errors")
        return 1

if __name__ == '__main__':
    sys.exit(run_suite())
