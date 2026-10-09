"""
Unit tests for BinanceClient and exchange utility functions.
"""

import unittest

try:
    import ccxt
    from src.exchanges.binance_client import BinanceClient
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


class TestBinanceClient(unittest.TestCase):
    def setUp(self):
        if HAS_CCXT:
            self.client = BinanceClient(dry_run=True, testnet=True)

    @unittest.skipUnless(HAS_CCXT, "ccxt not installed in local environment")
    def test_timeframe_to_ms(self):
        self.assertEqual(BinanceClient.timeframe_to_ms('1m'), 60000)
        self.assertEqual(BinanceClient.timeframe_to_ms('5m'), 300000)
        self.assertEqual(BinanceClient.timeframe_to_ms('1h'), 3600000)
        self.assertEqual(BinanceClient.timeframe_to_ms('1d'), 86400000)

    @unittest.skipUnless(HAS_CCXT, "ccxt not installed in local environment")
    def test_dry_run_balance(self):
        balance = self.client.get_available_balance('USDT')
        self.assertEqual(balance, 10000.0)

    @unittest.skipUnless(HAS_CCXT, "ccxt not installed in local environment")
    def test_dry_run_limit_buy(self):
        result = self.client.place_limit_buy('SOL/USDT', limit_price=150.0, quote_amount=300.0)
        self.assertTrue(result['dry_run'])
        self.assertEqual(result['status'], 'closed')
        self.assertEqual(result['limit_price'], 150.0)
        self.assertEqual(result['filled_amount'], 2.0)

    @unittest.skipUnless(HAS_CCXT, "ccxt not installed in local environment")
    def test_dry_run_oco_sell(self):
        result = self.client.place_oco_sell('SOL/USDT', base_amount=2.0, tp_price=160.0, sl_price=145.0)
        self.assertTrue(result['dry_run'])
        self.assertEqual(result['take_profit'], 160.0)
        self.assertEqual(result['stop_loss'], 145.0)

    @unittest.skipUnless(HAS_CCXT, "ccxt not installed in local environment")
    def test_dry_run_order_status(self):
        status = self.client.check_order_status('SOL/USDT', 'dry-run-limit-buy')
        self.assertEqual(status['status'], 'closed')


if __name__ == "__main__":
    unittest.main()

