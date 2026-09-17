import requests
import os

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN', '')   # from BotFather
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')      # from userinfobot

def notify_signal_with_liq(signal, daily_trades, liq, level_text='',symbol=''):
    emoji = '🟢' if signal['action'] == 'BUY' else '🔴'
    liq_emoji = '🐻' if liq and liq['bias'] == 'BEARISH' else '🐂' if liq and liq['bias'] == 'BULLISH' else '⚖️'

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
    conf_str = f"{signal['confidence']:.0%}" + (f" (Bullish: {p_bull:.0%})" if p_bull else "")

    msg = (
        f"{emoji} <b>{signal['action']} Signal — {symbol}</b>\n\n"
        f"💰 Entry:       <b>{signal['entry']}</b>\n"
        f"🎯 Take Profit: <b>{signal['take_profit']}</b>{fib_str}\n"
        f"🛑 Stop Loss:   <b>{signal['stop_loss']}</b>\n"
        f"📊 Confidence:  <b>{conf_str}</b>\n"
        f"🔢 Trade:       <b>{daily_trades}/8</b>"
        f"{liq_section}"
        f"{level_text}"
    )
    send_message(msg)

def send_message(text: str):
    try:
        url  = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        data = {
            'chat_id':    TELEGRAM_CHAT_ID,
            'text':       text,
            'parse_mode': 'HTML'
        }
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")


def notify_signal(signal, daily_trades):
    if signal['action'] == 'BUY':
        emoji = '🟢'
    else:
        emoji = '🔴'

    msg = (
        f"{emoji} <b>{signal['action']} Signal — ZEC/USDT</b>\n\n"
        f"💰 Entry:       <b>{signal['entry']}</b>\n"
        f"🎯 Take Profit: <b>{signal['take_profit']}</b>\n"
        f"🛑 Stop Loss:   <b>{signal['stop_loss']}</b>\n"
        f"📊 Confidence:  <b>{signal['confidence']:.0%}</b>\n"
        f"🔢 Trade:       <b>{daily_trades}/8</b>"
    )
    send_message(msg)


def notify_retrain(candles, avg_conf, max_conf):
    msg = (
        f"🤖 <b>Model Retrained</b>\n\n"
        f"📦 Candles:     <b>{candles}</b>\n"
        f"📈 Avg Conf:    <b>{avg_conf:.0%}</b>\n"
        f"🔝 Max Conf:    <b>{max_conf:.0%}</b>"
    )
    send_message(msg)


def notify_daily_limit():
    send_message("⛔ <b>Max trades reached for today (8/8)</b>")


def notify_loss_limit():
    send_message("🚨 <b>Daily loss limit hit (-1.5%), stopping for today</b>")


def notify_error(error: str):
    send_message(f"❌ <b>Bot Error</b>\n<code>{error}</code>")


def notify_start(symbol: str):
    send_message(f"✅ <b>{symbol} Trading Bot Started</b>")


def notify_stop():
    send_message("🛑 <b>Bot stopped by user</b>")