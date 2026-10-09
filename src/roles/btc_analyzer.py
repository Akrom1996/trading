"""
BTC Macro Trend Calculation Engine.
Monitors BTC multi-timeframe trend, moving averages, RSI, order book depth, and liquidation bias.
Publishes regime status to shared storage for altcoin strategy synchronization.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import ta

from config.settings import settings
from src.exchanges.binance_client import BinanceClient, get_binance_client
from src.indicators.order_book import calculate_depth_delta
from src.indicators.heatmap import get_liq_data
from src.services.redis_cache import set_shared_value
from src.services.telegram_notifier import notify_btc_regime, notify_error


class BTCAnalyzer:
    """
    BTC Macro Trend calculation engine.
    """

    def __init__(self, client: Optional[BinanceClient] = None):
        self.client = client or get_binance_client()
        self.symbol = "BTC/USDT"
        self.last_regime: Optional[str] = None
        self.last_alert_time: Optional[datetime] = None

    def calculate_macro_regime(self) -> Dict[str, Any]:
        """
        Analyzes 1h candles, order book, and liquidations for BTC.
        """
        # 1. Fetch 1h candles for macro trend
        df_1h = self.client.fetch_ohlcv(self.symbol, timeframe='1h', limit=250)
        if len(df_1h) < 200:
            # Fallback to 5m if 1h has insufficient history
            df_1h = self.client.fetch_ohlcv(self.symbol, timeframe='5m', limit=500)

        close = df_1h['close']
        current_price = float(close.iloc[-1])

        ema_20 = float(ta.trend.ema_indicator(close, window=20).iloc[-1])
        ema_50 = float(ta.trend.ema_indicator(close, window=50).iloc[-1])
        ema_200 = float(ta.trend.ema_indicator(close, window=200).iloc[-1])
        rsi_1h = float(ta.momentum.rsi(close, window=14).iloc[-1])

        # 2. Fetch order book depth
        ob_bias = "NEUTRAL"
        imbalance = 0.0
        try:
            ob = self.client.fetch_order_book(self.symbol, limit=50)
            ob_metrics = calculate_depth_delta(ob.get('bids', []), ob.get('asks', []), depth_levels=20)
            ob_bias = ob_metrics.get('bias', 'NEUTRAL')
            imbalance = ob_metrics.get('imbalance_ratio', 0.0)
        except Exception as e:
            print(f"[BTCAnalyzer] Order book fetch error: {e}")

        # 3. Liquidation bias
        liq_bias = "NEUTRAL"
        try:
            liq_info, _ = get_liq_data('BTC', current_price)
            if liq_info:
                liq_bias = liq_info.get('bias', 'NEUTRAL')
        except Exception as e:
            print(f"[BTCAnalyzer] Liquidation fetch error: {e}")

        # 4. Synthesize Macro Regime
        bullish_score = 0
        bearish_score = 0

        # Moving average trend
        if current_price > ema_50:
            bullish_score += 1
        else:
            bearish_score += 1

        if ema_50 > ema_200:
            bullish_score += 2
        else:
            bearish_score += 2

        # RSI Momentum
        if rsi_1h > 55:
            bullish_score += 1
        elif rsi_1h < 45:
            bearish_score += 1

        # Order book delta
        if ob_bias == "BULLISH":
            bullish_score += 1
        elif ob_bias == "BEARISH":
            bearish_score += 1

        # Liquidation
        if liq_bias == "BULLISH":
            bullish_score += 1
        elif liq_bias == "BEARISH":
            bearish_score += 1

        if bullish_score >= 4 and bullish_score > bearish_score + 1:
            bias = "BULLISH"
            summary = "Macro uptrend intact. Price above EMAs with favorable momentum."
        elif bearish_score >= 4 and bearish_score > bullish_score + 1:
            bias = "BEARISH"
            summary = "Macro downtrend pressure. Price below key EMAs with selling imbalance."
        else:
            bias = "NEUTRAL"
            summary = "Consolidation or chop. Directional bias is mixed."

        regime_data = {
            'symbol': self.symbol,
            'price': current_price,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'ema_200': ema_200,
            'rsi_1h': rsi_1h,
            'ob_bias': ob_bias,
            'imbalance': imbalance,
            'liq_bias': liq_bias,
            'bias': bias,
            'summary': summary,
            'updated_at': datetime.now().isoformat(),
        }

        # Cache in shared storage for coin traders to reference
        set_shared_value("btc:macro_regime", regime_data, expire_seconds=1800)
        return regime_data

    def run(self, poll_interval_seconds: int = 900):
        """
        Runs the BTC macro analyzer daemon loop.
        """
        print(f"[BTCAnalyzer] 🚀 Starting BTC Macro Trend Engine (polling every {poll_interval_seconds}s)")
        while True:
            try:
                regime = self.calculate_macro_regime()
                now = datetime.now()
                bias = regime['bias']

                print(
                    f"[BTCAnalyzer] [{now.strftime('%H:%M')}] BTC: ${regime['price']:,.2f} | "
                    f"Regime: {bias} | RSI: {regime['rsi_1h']:.1f} | OB: {regime['ob_bias']}"
                )

                # Send Telegram alert on regime transition or every 4 hours
                regime_changed = self.last_regime is not None and self.last_regime != bias
                time_for_update = self.last_alert_time is None or (now - self.last_alert_time).total_seconds() > 14400

                if regime_changed or time_for_update:
                    notify_btc_regime(regime)
                    self.last_regime = bias
                    self.last_alert_time = now

            except Exception as e:
                print(f"[BTCAnalyzer] Error in analysis loop: {e}")
                notify_error(f"BTC Analyzer Error: {e}")

            time.sleep(poll_interval_seconds)


def run_btc_analyzer():
    analyzer = BTCAnalyzer()
    analyzer.run()

