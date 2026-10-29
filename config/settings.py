"""
Application configuration management using Pydantic and environment variables.
"""

import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

        # Telegram Configuration
        TELEGRAM_TOKEN: str = Field(default="", description="Telegram Bot Token")
        TELEGRAM_CHAT_ID: Optional[str] = Field(default=None, description="Main Telegram channel / chat ID")
        TELEGRAM_ERROR_CHAT_ID: Optional[str] = Field(default=None, description="Telegram error / admin chat ID")

        # Binance Configuration
        BINANCE_API_KEY: Optional[str] = Field(default=None, description="Binance API Key")
        BINANCE_API_SECRET: Optional[str] = Field(default=None, description="Binance API Secret")
        BINANCE_TESTNET: bool = Field(default=True, description="Use Binance Spot Testnet")
        DRY_RUN: bool = Field(default=True, description="Simulate orders without placing on exchange")

        # Trading Pair & Role
        SYMBOL: str = Field(default="SOL/USDT", description="Trading pair symbol (e.g. SOL/USDT)")
        ROLE: str = Field(default="coin_trader", description="Active bot role: coin_trader, btc_analyzer, reporter")

        # Trading Execution Parameters
        QUOTE_AMOUNT_PER_TRADE: float = Field(default=20.0, description="USDT allocated per trade entry")
        ENTRY_OFFSET_PCT: float = Field(default=0.003, description="Limit buy offset below current market price (0.3%)")
        SL_COOLDOWN_MINUTES: int = Field(default=30, description="Cooldown period after a Stop Loss hit in minutes")
        TP_COOLDOWN_MINUTES: int = Field(default=10, description="Cooldown period after a Take Profit hit in minutes")
        SELL_ALERT_COOLDOWN_MINUTES: float = Field(default=30.0, description="Cooldown period between Telegram SELL alerts in minutes")
        MAX_TRADES_PER_DAY: int = Field(default=10, description="Maximum entries allowed per 24h cycle")
        MAX_RISK_PER_TRADE: float = Field(default=0.03, description="Maximum risk per trade (fraction)")
        MAX_DAILY_LOSS: float = Field(default=6.0, description="Maximum daily loss percentage before stopping")

        # Profit Lock & Trailing Protection
        PROFIT_LOCK_TRIGGER_PCT: float = Field(default=0.01, description="Price gain above entry to trigger profit lock (e.g. 0.01 for +1%)")
        PROFIT_LOCK_SL_PCT: float = Field(default=0.01, description="New stop loss percentage above entry price when profit is locked (e.g. 0.01 for +1%)")
        BTC_FILTER_ENABLED: bool = Field(default=False, description="Skip altcoin BUY entries if BTC macro regime is BEARISH")

        # Model & Market Dynamics
        TRAIN_CANDLES: int = Field(default=2880, description="Historical candles for training (10 days of 5m)")
        LIVE_CANDLES: int = Field(default=1000, description="Candles fetched for live inference")
        MAX_MODEL_AGE_HRS: int = Field(default=12, description="Max model age in hours before retraining")
        MAX_PYRAMID_POSITIONS: int = Field(default=6, description="Max stacked positions per symbol")
        PYRAMID_TRIGGER_PCT: float = Field(default=0.002, description="Threshold near TP to evaluate pyramid entry")
        PENDING_FILL_TIMEOUT_MINUTES: int = Field(default=15, description="Timeout before canceling unfilled limit order")

        # Pyramiding / risk / model gates
        PYRAMID_SIZE_MULT: float = Field(default=0.5, description="Pyramid rung size as a fraction of QUOTE_AMOUNT_PER_TRADE")
        PYRAMID_CHOP_BLOCK_MINUTES: int = Field(default=30, description="Block pyramids for this long after any CHOP signal")
        PYRAMID_REQUIRE_LIQ_FEED: bool = Field(default=True, description="Only pyramid while the liquidation feed is connected")
        MAX_OPEN_RISK_PCT: float = Field(default=0.03, description="Max summed stop distance (size-weighted) across open positions")
        FEE_ROUND_TRIP: float = Field(default=0.002, description="Round-trip trading fee fraction used in EV calculations")
        MIN_EDGE_PCT: float = Field(default=0.003, description="Minimum expected value (net of fees) required to enter")
        SIGNAL_MARGIN: float = Field(default=0.10, description="Required P(bullish) margin above the training base rate")
        LIQ_MIN_EVENTS: int = Field(default=3, description="Min liquidation events in window before a liq bias is trusted")
        LIQ_MIN_NOTIONAL_USD: float = Field(default=20000.0, description="Min liquidation notional in window before a liq bias is trusted")
        CHOP_ALERT_COOLDOWN_MINUTES: Optional[int] = Field(default=None, description="Cooldown between CHOP alerts (falls back to SELL_ALERT_COOLDOWN_MINUTES)")

        # Database & Cache Persistence
        DB_PATH: str = Field(default="data/bot_data.db", description="Path to SQLite database")
        REDIS_URL: Optional[str] = Field(default=None, description="Redis connection URL (optional)")
        REDIS_HOST: str = Field(default="localhost", description="Redis host")
        REDIS_PORT: int = Field(default=6379, description="Redis port")
        STATE_DIR: str = Field(default="state", description="Directory where model artifacts are saved")

except ImportError:
    # Graceful fallback when pydantic / pydantic-settings is not installed locally
    def _parse_bool(val: Optional[str], default: bool = False) -> bool:
        if val is None:
            return default
        return val.strip().lower() in ("true", "1", "yes", "on")

    class Settings:
        def __init__(self):
            # Load .env manually if exists
            if os.path.exists(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v

            self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
            self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
            self.TELEGRAM_ERROR_CHAT_ID = os.getenv("TELEGRAM_ERROR_CHAT_ID")

            self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
            self.BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
            self.BINANCE_TESTNET = _parse_bool(os.getenv("BINANCE_TESTNET"), True)
            self.DRY_RUN = _parse_bool(os.getenv("DRY_RUN"), True)

            self.SYMBOL = os.getenv("SYMBOL", "SOL/USDT")
            self.ROLE = os.getenv("ROLE", "coin_trader")

            self.QUOTE_AMOUNT_PER_TRADE = float(os.getenv("QUOTE_AMOUNT_PER_TRADE", "20.0"))
            self.ENTRY_OFFSET_PCT = float(os.getenv("ENTRY_OFFSET_PCT", "0.003"))
            self.SL_COOLDOWN_MINUTES = int(os.getenv("SL_COOLDOWN_MINUTES", "30"))
            self.TP_COOLDOWN_MINUTES = int(os.getenv("TP_COOLDOWN_MINUTES", "10"))
            self.SELL_ALERT_COOLDOWN_MINUTES = float(os.getenv("SELL_ALERT_COOLDOWN_MINUTES", "30.0"))
            self.MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "10"))
            self.MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "0.03"))
            self.MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "6.0"))

            self.PROFIT_LOCK_TRIGGER_PCT = float(os.getenv("PROFIT_LOCK_TRIGGER_PCT", "0.01"))
            self.PROFIT_LOCK_SL_PCT = float(os.getenv("PROFIT_LOCK_SL_PCT", "0.01"))
            self.BTC_FILTER_ENABLED = _parse_bool(os.getenv("BTC_FILTER_ENABLED"), False)

            self.TRAIN_CANDLES = int(os.getenv("TRAIN_CANDLES", "2880"))
            self.LIVE_CANDLES = int(os.getenv("LIVE_CANDLES", "1000"))
            self.MAX_MODEL_AGE_HRS = int(os.getenv("MAX_MODEL_AGE_HRS", "12"))
            self.MAX_PYRAMID_POSITIONS = int(os.getenv("MAX_PYRAMID_POSITIONS", "6"))
            self.PYRAMID_TRIGGER_PCT = float(os.getenv("PYRAMID_TRIGGER_PCT", "0.002"))
            self.PENDING_FILL_TIMEOUT_MINUTES = int(os.getenv("PENDING_FILL_TIMEOUT_MINUTES", "15"))

            self.PYRAMID_SIZE_MULT = float(os.getenv("PYRAMID_SIZE_MULT", "0.5"))
            self.PYRAMID_CHOP_BLOCK_MINUTES = int(os.getenv("PYRAMID_CHOP_BLOCK_MINUTES", "30"))
            self.PYRAMID_REQUIRE_LIQ_FEED = _parse_bool(os.getenv("PYRAMID_REQUIRE_LIQ_FEED"), True)
            self.MAX_OPEN_RISK_PCT = float(os.getenv("MAX_OPEN_RISK_PCT", "0.03"))
            self.FEE_ROUND_TRIP = float(os.getenv("FEE_ROUND_TRIP", "0.002"))
            self.MIN_EDGE_PCT = float(os.getenv("MIN_EDGE_PCT", "0.003"))
            self.SIGNAL_MARGIN = float(os.getenv("SIGNAL_MARGIN", "0.10"))
            self.LIQ_MIN_EVENTS = int(os.getenv("LIQ_MIN_EVENTS", "3"))
            self.LIQ_MIN_NOTIONAL_USD = float(os.getenv("LIQ_MIN_NOTIONAL_USD", "20000"))
            _chop_cd = os.getenv("CHOP_ALERT_COOLDOWN_MINUTES")
            self.CHOP_ALERT_COOLDOWN_MINUTES = int(_chop_cd) if _chop_cd else None

            self.DB_PATH = os.getenv("DB_PATH", "data/bot_data.db")
            self.REDIS_URL = os.getenv("REDIS_URL")
            self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
            self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
            self.STATE_DIR = os.getenv("STATE_DIR", "state")

settings = Settings()

