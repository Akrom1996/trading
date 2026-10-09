"""
Unit tests for configuration settings.
"""

import unittest
from config.settings import settings, Settings


class TestSettings(unittest.TestCase):
    def test_default_settings(self):
        self.assertIsNotNone(settings.SYMBOL)
        self.assertIsInstance(settings.QUOTE_AMOUNT_PER_TRADE, float)
        self.assertIsInstance(settings.MAX_TRADES_PER_DAY, int)
        self.assertIsInstance(settings.ENTRY_OFFSET_PCT, float)
        self.assertIsInstance(settings.SL_COOLDOWN_MINUTES, float)
        self.assertIsInstance(settings.TP_COOLDOWN_MINUTES, float)
        self.assertIn(settings.ROLE, ["coin_trader", "btc_analyzer", "reporter"])

    def test_custom_settings(self):
        test_settings = Settings()
        self.assertTrue(hasattr(test_settings, "DRY_RUN"))
        self.assertTrue(hasattr(test_settings, "DB_PATH"))


if __name__ == "__main__":
    unittest.main()

