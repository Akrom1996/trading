"""
Telegram notification dispatcher for signals, trade executions, daily limits, and errors.
"""

from typing import Optional, Dict, Any

try:
    import requests
except ImportError:
    requests = None

from config.settings import settings


class TelegramNotifier:
    """
    Dispatcher for formatting and sending alerts to Telegram channels.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        error_chat_id: Optional[str] = None,
    ):
        self.token = token or settings.TELEGRAM_TOKEN
        self.chat_id = chat_id or settings.TELEGRAM_CHAT_ID
        self.error_chat_id = error_chat_id or settings.TELEGRAM_ERROR_CHAT_ID

    def send_message(self, text: str, chat_id: Optional[str] = None) -> bool:
        """Sends an HTML-formatted message to Telegram."""
        target_chat = chat_id or self.chat_id
        if not self.token or not target_chat:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': target_chat,
                'text': text,
                'parse_mode': 'HTML',
            }
            res = requests.post(url, data=data, timeout=5)
            return res.status_code == 200
        except Exception as e:
            print(f"[TelegramNotifier] Error sending message: {e}")
            return False

    def notify_signal_with_liq(
        self,
        signal: Dict[str, Any],
        daily_trades: int,
        liq: Optional[Dict[str, Any]] = None,
        level_text: str = '',
        symbol: str = '',
    ):
        emoji = '🟢' if signal.get('action') == 'BUY' else '🔴'
        liq_emoji = '🐻' if liq and liq.get('bias') == 'BEARISH' else '🐂' if liq and liq.get('bias') == 'BULLISH' else '⚖️'

        liq_section = ""
        if liq:
            liq_section = (
                f"\n\n📊 <b>Liquidation Data:</b>\n"
                f"  {liq_emoji} Bias: <b>{liq['bias']}</b>\n"
                f"  📉 Long liq:  <b>${liq['long_liq_usd']:,.0f}</b>\n"
                f"  📈 Short liq: <b>${liq['short_liq_usd']:,.0f}</b>\n"
                f"  💬 {liq['comment']}"
            )

        fib_target = signal.get('fib_target_pct')
        fib_str = f" (+{fib_target*100:.0f}% Fib)" if fib_target else ""
        p_bull = signal.get('p_bullish')
        conf_str = f"{signal.get('confidence', 0):.0%}" + (f" (Bullish: {p_bull:.0%})" if p_bull else "")

        msg = (
            f"{emoji} <b>{signal.get('action')} Signal — {symbol}</b>\n\n"
            f"💰 Entry:       <b>{signal.get('entry')}</b>\n"
            f"🎯 Take Profit: <b>{signal.get('take_profit')}</b>{fib_str}\n"
            f"🛑 Stop Loss:   <b>{signal.get('stop_loss')}</b>\n"
            f"📊 Confidence:  <b>{conf_str}</b>\n"
            f"🔢 Trade:       <b>{daily_trades}/{settings.MAX_TRADES_PER_DAY}</b>"
            f"{liq_section}"
            f"{level_text}"
        )
        self.send_message(msg)

    def notify_signal(self, signal: Dict[str, Any], daily_trades: int, symbol: str = 'SOL/USDT'):
        emoji = '🟢' if signal.get('action') == 'BUY' else '🔴'
        msg = (
            f"{emoji} <b>{signal.get('action')} Signal — {symbol}</b>\n\n"
            f"💰 Entry:       <b>{signal.get('entry')}</b>\n"
            f"🎯 Take Profit: <b>{signal.get('take_profit')}</b>\n"
            f"🛑 Stop Loss:   <b>{signal.get('stop_loss')}</b>\n"
            f"📊 Confidence:  <b>{signal.get('confidence', 0):.0%}</b>\n"
            f"🔢 Trade:       <b>{daily_trades}/{settings.MAX_TRADES_PER_DAY}</b>"
        )
        self.send_message(msg)

    def notify_retrain(self, candles: int, avg_conf: float, max_conf: float, symbol: str = ''):
        tag = f" — {symbol}" if symbol else ""
        msg = (
            f"🤖 <b>Model Retrained{tag}</b>\n\n"
            f"📦 Candles:     <b>{candles}</b>\n"
            f"📈 Avg Conf:    <b>{avg_conf:.0%}</b>\n"
            f"🔝 Max Conf:    <b>{max_conf:.0%}</b>"
        )
        self.send_message(msg, self.error_chat_id)

    def notify_daily_limit(self, symbol: str = ''):
        tag = f" for {symbol}" if symbol else ""
        self.send_message(
            f"⛔ <b>Max trades reached for today ({settings.MAX_TRADES_PER_DAY}/{settings.MAX_TRADES_PER_DAY}){tag}</b>",
            self.error_chat_id,
        )

    def notify_loss_limit(self, symbol: str):
        self.send_message(
            f"🚨 <b>Daily loss limit hit (-{settings.MAX_DAILY_LOSS}%), stopping for today for {symbol}</b>",
            self.error_chat_id,
        )

    def notify_error(self, error: str):
        self.send_message(f"❌ <b>Bot Error</b>\n<code>{error}</code>", self.error_chat_id)

    def notify_start(self, symbol: str):
        self.send_message(f"✅ <b>{symbol} Trading Bot Started</b>", self.error_chat_id)

    def notify_stop(self, symbol: str = ''):
        tag = f" for {symbol}" if symbol else ""
        self.send_message(f"🛑 <b>Bot stopped by user{tag}</b>", self.error_chat_id)

    def notify_btc_regime(self, regime_data: Dict[str, Any]):
        bias = regime_data.get('bias', 'NEUTRAL')
        icon = "🟢" if bias == "BULLISH" else "🔴" if bias == "BEARISH" else "⚖️"
        msg = (
            f"{icon} <b>BTC Macro Regime Update: {bias}</b>\n\n"
            f"💰 Price:       <b>${regime_data.get('price', 0):,.2f}</b>\n"
            f"📈 EMA 50/200:  <b>${regime_data.get('ema_50', 0):,.2f} / ${regime_data.get('ema_200', 0):,.2f}</b>\n"
            f"⚡ 1h RSI:      <b>{regime_data.get('rsi_1h', 0):.1f}</b>\n"
            f"📊 Order Book:  <b>{regime_data.get('ob_bias', 'N/A')} ({regime_data.get('imbalance', 0):+.1%})</b>"
        )
        self.send_message(msg, self.error_chat_id)


# ── Global Default Notifier & Functional API ─────────────────────
_default_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    global _default_notifier
    if _default_notifier is None:
        _default_notifier = TelegramNotifier()
    return _default_notifier


def send_message(text: str, chat_id: Optional[str] = None) -> bool:
    return get_notifier().send_message(text, chat_id)

def notify_signal_with_liq(signal, daily_trades, liq, level_text='', symbol=''):
    get_notifier().notify_signal_with_liq(signal, daily_trades, liq, level_text, symbol)

def notify_signal(signal, daily_trades, symbol='SOL/USDT'):
    get_notifier().notify_signal(signal, daily_trades, symbol)

def notify_retrain(candles, avg_conf, max_conf, symbol=''):
    get_notifier().notify_retrain(candles, avg_conf, max_conf, symbol)

def notify_daily_limit(symbol=''):
    get_notifier().notify_daily_limit(symbol)

def notify_loss_limit(symbol: str):
    get_notifier().notify_loss_limit(symbol)

def notify_error(error: str):
    get_notifier().notify_error(error)

def notify_start(symbol: str):
    get_notifier().notify_start(symbol)

def notify_stop(symbol: str = ''):
    get_notifier().notify_stop(symbol)

def notify_btc_regime(regime_data: Dict[str, Any]):
    get_notifier().notify_btc_regime(regime_data)


def send_daily_pnl_report():
    """Generates and sends automated daily PnL breakdown across all coins."""
    from src.services.redis_cache import get_daily_summary
    summary, overall_pnl, overall_trades = get_daily_summary()

    if not summary:
        send_message("📊 <b>Automated Daily Report</b>\n\nNo trades closed today.")
        return

    status_icon = "🟢" if overall_pnl >= 0 else "🔴"
    msg = (
        f"📊 <b>Automated Daily Report</b>\n"
        f"Overall Daily PnL: <b>{status_icon} {overall_pnl:+.2f}%</b>\n"
        f"Total Trades Closed: <b>{overall_trades}</b>\n\n"
        f"<b>Coin Breakdown:</b>\n"
    )

    for item in summary:
        coin_icon = "📈" if item["pnl_pct"] >= 0 else "📉"
        win_rate = ((item["wins"] / item["trades"]) * 100) if item["trades"] > 0 else 0
        msg += (
            f"{coin_icon} <b>{item['symbol']}</b>: <b>{item['pnl_pct']:+.2f}%</b> "
            f"({item['trades']} trades | WR: {win_rate:.0f}%)\n"
        )

    send_message(msg)


def run_daily_reporter():
    """Service loop scheduling the daily report for 23:59."""
    import time
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(send_daily_pnl_report, "cron", hour=23, minute=59)
    scheduler.start()
    print("📊 Automated Daily Report Service Running (Scheduled for 23:59)...")
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

