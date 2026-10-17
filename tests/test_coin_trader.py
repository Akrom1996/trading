"""
Unit tests for coin_trader order placement and BTC trend logic.
"""

import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# Ensure optional heavy dependencies are stubbed if not installed in local environment
for mod in ['ccxt', 'pandas', 'ta', 'sklearn', 'xgboost']:
    sys.modules.setdefault(mod, MagicMock())

from config.settings import settings
from src.roles.coin_trader import open_position, get_decimal_places
from src.services.telegram_notifier import TelegramNotifier


class TestCoinTrader(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, 12, 0, 0)
        self.symbol = "SOL/USDT"

    @patch("src.roles.coin_trader.place_market_buy")
    def test_open_position_market(self, mock_market_buy):
        mock_market_buy.return_value = {
            "id": "ord-mkt-1",
            "status": "closed",
            "filled_amount": 1.5,
            "avg_price": 100.0,
        }
        with patch("src.roles.coin_trader.protect_position") as mock_protect:
            pos = open_position(
                price=100.0,
                tp_pct=0.03,
                sl_pct=0.02,
                now=self.now,
                label="fresh market entry",
                symbol=self.symbol,
                order_type="market",
            )
            mock_market_buy.assert_called_once_with(self.symbol, settings.QUOTE_AMOUNT_PER_TRADE)
            self.assertEqual(pos["order_type"], "market")
            self.assertEqual(pos["entry"], 100.0)
            self.assertEqual(pos["take_profit"], 103.0)
            self.assertEqual(pos["stop_loss"], 98.0)
            self.assertEqual(pos["status"], "open")
            mock_protect.assert_called_once()

    @patch("src.roles.coin_trader.place_limit_buy")
    def test_open_position_limit(self, mock_limit_buy):
        mock_limit_buy.return_value = {
            "id": "ord-lmt-1",
            "status": "open",
            "filled_amount": 0.0,
        }
        limit_price = round(100.0 * (1 - settings.ENTRY_OFFSET_PCT), 2)
        pos = open_position(
            price=limit_price,
            tp_pct=0.03,
            sl_pct=0.02,
            now=self.now,
            label="fresh limit entry",
            symbol=self.symbol,
            order_type="limit",
        )
        mock_limit_buy.assert_called_once_with(self.symbol, limit_price, settings.QUOTE_AMOUNT_PER_TRADE)
        decimals = get_decimal_places(limit_price)
        self.assertEqual(pos["order_type"], "limit")
        self.assertEqual(pos["entry"], limit_price)
        self.assertEqual(pos["take_profit"], round(limit_price * 1.03, decimals))
        self.assertEqual(pos["stop_loss"], round(limit_price * 0.98, decimals))
        self.assertEqual(pos["status"], "pending_fill")

    @patch("src.roles.coin_trader.place_limit_buy")
    def test_open_position_pyramid_defaults_to_limit(self, mock_limit_buy):
        mock_limit_buy.return_value = {
            "id": "ord-pyr-1",
            "status": "open",
            "filled_amount": 0.0,
        }
        pos = open_position(
            price=99.7,
            tp_pct=0.03,
            sl_pct=0.02,
            now=self.now,
            label="pyramid entry",
            symbol=self.symbol,
            is_pyramid=True,
        )
        self.assertEqual(pos["order_type"], "limit")
        mock_limit_buy.assert_called_once()

    def test_btc_bias_decision_logic(self):
        current_price = 150.0
        decimals = get_decimal_places(current_price)
        effective_tp = 0.04
        sl_pct = 0.02

        # 1. Bullish regime -> market buy at current_price
        btc_bias = "BULLISH"
        if btc_bias == "BULLISH":
            entry_price = current_price
            order_type = "market"
        else:
            entry_price = round(current_price * (1 - settings.ENTRY_OFFSET_PCT), decimals)
            order_type = "limit"

        self.assertEqual(order_type, "market")
        self.assertEqual(entry_price, 150.0)
        tp = round(entry_price * (1 + effective_tp), decimals)
        sl = round(entry_price * (1 - sl_pct), decimals)
        self.assertEqual(tp, 156.0)
        self.assertEqual(sl, 147.0)

        # 2. Neutral regime -> limit buy using ENTRY_OFFSET_PCT
        btc_bias = "NEUTRAL"
        if btc_bias == "BULLISH":
            entry_price = current_price
            order_type = "market"
        else:
            entry_price = round(current_price * (1 - settings.ENTRY_OFFSET_PCT), decimals)
            order_type = "limit"

        expected_limit_price = round(150.0 * (1 - settings.ENTRY_OFFSET_PCT), decimals)
        self.assertEqual(order_type, "limit")
        self.assertEqual(entry_price, expected_limit_price)
        self.assertLess(entry_price, current_price)
        tp = round(entry_price * (1 + effective_tp), decimals)
        sl = round(entry_price * (1 - sl_pct), decimals)
        self.assertEqual(tp, round(expected_limit_price * 1.04, decimals))
        self.assertEqual(sl, round(expected_limit_price * 0.98, decimals))

    def test_telegram_signal_order_type_display(self):
        notifier = TelegramNotifier()
        sent_messages = []
        notifier.send_message = lambda msg, chat_id=None: sent_messages.append(msg)

        signal_mkt = {
            "action": "BUY",
            "entry": 150.0,
            "take_profit": 156.0,
            "stop_loss": 147.0,
            "order_type": "market",
            "confidence": 0.85,
        }
        notifier.notify_signal_with_liq(signal_mkt, daily_trades=1, symbol=self.symbol)
        self.assertTrue(any("BUY Signal (MARKET)" in m for m in sent_messages))

        signal_lmt = {
            "action": "BUY",
            "entry": 149.55,
            "take_profit": 155.53,
            "stop_loss": 146.56,
            "order_type": "limit",
            "confidence": 0.85,
        }
        notifier.notify_signal_with_liq(signal_lmt, daily_trades=2, symbol=self.symbol)
        self.assertTrue(any("BUY Signal (LIMIT)" in m for m in sent_messages))


if __name__ == "__main__":
    unittest.main()
