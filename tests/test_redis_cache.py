"""
Unit tests for shared state storage and SQLite persistence.
"""

import os
import tempfile
import unittest
from src.services.redis_cache import StateStorage


class TestStateStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_bot.db")
        self.storage = StateStorage(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_bot_state(self):
        symbol = "TEST/USDT"
        open_positions = [
            {
                "action": "BUY",
                "entry": 100.0,
                "take_profit": 103.0,
                "stop_loss": 98.0,
                "status": "open",
            }
        ]
        daily_trades = 2
        daily_pnl = 1.5
        day = 15

        self.storage.save_bot_state(symbol, open_positions, daily_trades, daily_pnl, day)

        loaded_positions, loaded_trades, loaded_pnl, loaded_day = self.storage.load_bot_state(symbol)
        self.assertEqual(len(loaded_positions), 1)
        self.assertEqual(loaded_positions[0]["entry"], 100.0)
        self.assertEqual(loaded_trades, 2)
        self.assertAlmostEqual(loaded_pnl, 1.5)
        self.assertEqual(loaded_day, 15)

    def test_record_closed_trade_and_summary(self):
        symbol = "TEST/USDT"
        self.storage.record_closed_trade(symbol, "BUY", 100.0, 105.0, 5.0, "TP Hit (Candle High)")
        self.storage.record_closed_trade(symbol, "BUY", 100.0, 98.0, -2.0, "SL Hit (Candle Low)")

        summary, overall_pnl, overall_trades = self.storage.get_daily_summary()
        self.assertEqual(overall_trades, 2)
        self.assertAlmostEqual(overall_pnl, 3.0)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["symbol"], symbol)
        self.assertEqual(summary[0]["wins"], 1)

    def test_cooldown_detection(self):
        symbol = "TEST/USDT"
        self.storage.record_closed_trade(symbol, "BUY", 100.0, 98.0, -2.0, "SL Hit (Candle Low)")
        sl_time = self.storage.get_last_sl_time(symbol)
        self.assertIsNotNone(sl_time)

        tp_time = self.storage.get_last_tp_time(symbol)
        self.assertIsNone(tp_time)


if __name__ == "__main__":
    unittest.main()

