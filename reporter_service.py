import os
import time
from apscheduler.schedulers.background import BackgroundScheduler
from db import get_daily_summary
from telegram_bot import send_message


def send_daily_pnl_report():
    summary, overall_pnl, overall_trades = get_daily_summary()

    if not summary:
        send_message(
            "📊 <b>Automated Daily Report</b>\n\nNo trades closed today."
        )
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
        win_rate = (
            (item["wins"] / item["trades"]) * 100 if item["trades"] > 0 else 0
        )
        msg += (
            f"{coin_icon} <b>{item['symbol']}</b>: <b>{item['pnl_pct']:+.2f}%</b> "
            f"({item['trades']} trades | WR: {win_rate:.0f}%)\n"
        )

    send_message(msg)


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    # Trigger automatically every day at 23:59 local time
    scheduler.add_job(send_daily_pnl_report, "cron", hour=23, minute=59)
    scheduler.start()

    print("📊 Automated Daily Report Service Running...")
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()