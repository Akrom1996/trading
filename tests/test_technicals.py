"""
Unit tests for technical indicators and Fibonacci labeling.
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    import ta
    from src.indicators.technicals import (
        add_features,
        add_labels,
        FIBONACCI_LEVELS,
        FIB_TIER_NAMES,
        FEATURE_COLUMNS,
    )
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class TestTechnicals(unittest.TestCase):
    @unittest.skipUnless(HAS_DEPS, "numpy/pandas/ta not installed in local environment")
    def test_fibonacci_definitions(self):
        self.assertEqual(len(FIBONACCI_LEVELS), 6)
        self.assertIn(0.01, FIBONACCI_LEVELS)
        self.assertIn(0.13, FIBONACCI_LEVELS)
        self.assertIn(6, FIB_TIER_NAMES)

    @unittest.skipUnless(HAS_DEPS, "pandas/ta not installed in local environment")
    def test_add_features_and_labels(self):
        dates = pd.date_range('2026-01-01', periods=200, freq='5min')
        base_price = 100.0 + np.cumsum(np.random.randn(200) * 0.5)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': base_price,
            'high': base_price + np.random.uniform(0.1, 1.0, 200),
            'low': base_price - np.random.uniform(0.1, 1.0, 200),
            'close': base_price + np.random.uniform(-0.5, 0.5, 200),
            'volume': np.random.uniform(100, 1000, 200),
        })

        feat_df = add_features(df)
        self.assertGreater(len(feat_df), 0)
        for col in ['rsi', 'ema_9', 'ema_21', 'ema_50', 'bb_width', 'return_15m', 'range_15m']:
            self.assertIn(col, feat_df.columns)

        labeled_df = add_labels(feat_df, horizon=20, sl_pct=0.02)
        self.assertIn('label', labeled_df.columns)
        self.assertIn('max_gain', labeled_df.columns)
        self.assertTrue(all(labeled_df['label'].between(0, 6)))


if __name__ == "__main__":
    unittest.main()

