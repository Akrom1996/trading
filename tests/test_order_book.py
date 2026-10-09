"""
Unit tests for order book indicators and depth imbalance calculations.
"""

import unittest
from src.indicators.order_book import calculate_depth_delta, detect_walls


class TestOrderBook(unittest.TestCase):
    def test_calculate_depth_delta_bullish(self):
        bids = [[100.0, 10.0], [99.0, 15.0], [98.0, 20.0]]  # ~1000 + 1485 + 1960 = 4445
        asks = [[101.0, 2.0], [102.0, 3.0], [103.0, 1.0]]    # ~202 + 306 + 103 = 611

        result = calculate_depth_delta(bids, asks, depth_levels=3)
        self.assertEqual(result['bias'], 'BULLISH')
        self.assertGreater(result['bid_depth_usd'], result['ask_depth_usd'])
        self.assertGreater(result['imbalance_ratio'], 0.15)
        self.assertEqual(result['best_bid'], 100.0)
        self.assertEqual(result['best_ask'], 101.0)
        self.assertEqual(result['spread'], 1.0)

    def test_calculate_depth_delta_bearish(self):
        bids = [[100.0, 1.0], [99.0, 1.0]]
        asks = [[101.0, 20.0], [102.0, 30.0]]

        result = calculate_depth_delta(bids, asks, depth_levels=2)
        self.assertEqual(result['bias'], 'BEARISH')
        self.assertLess(result['imbalance_ratio'], -0.15)

    def test_empty_order_book(self):
        result = calculate_depth_delta([], [])
        self.assertEqual(result['bias'], 'NEUTRAL')
        self.assertEqual(result['delta_usd'], 0.0)

    def test_detect_walls(self):
        bids = [
            [100.0, 50.0],   # $5,000
            [99.0, 600.0],   # $59,400 -> Wall!
            [98.0, 40.0],    # $3,920
        ]
        asks = [
            [101.0, 50.0],
            [102.0, 50.0],
        ]
        walls = detect_walls(bids, asks, depth_levels=3, multiplier=2.0)
        self.assertGreaterEqual(len(walls['buy_walls']), 1)
        self.assertEqual(walls['buy_walls'][0]['price'], 99.0)


if __name__ == "__main__":
    unittest.main()

