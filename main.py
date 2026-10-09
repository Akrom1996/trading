"""
Main application entry point for crypto trading bot.
Dispatches to appropriate service role based on CLI arguments or environment variables.
"""

import argparse
import sys

from config.settings import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot Runner")
    parser.add_argument(
        "--role",
        type=str,
        default=settings.ROLE,
        choices=["coin_trader", "btc_analyzer", "reporter"],
        help="Service role to run: coin_trader, btc_analyzer, or reporter",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=settings.SYMBOL,
        help="Trading pair symbol (e.g. SOL/USDT)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    role = args.role.lower()
    symbol = args.symbol

    print("=" * 60)
    print(f"  CRYPTO TRADING BOT")
    print(f"  Role   : {role}")
    print(f"  Symbol : {symbol if role == 'coin_trader' else 'N/A'}")
    print(f"  Mode   : {'DRY RUN' if settings.DRY_RUN else 'LIVE'}")
    print("=" * 60)

    try:
        if role == "coin_trader":
            from src.roles.coin_trader import run_coin_trader
            run_coin_trader(symbol=symbol)
        elif role == "btc_analyzer":
            from src.roles.btc_analyzer import run_btc_analyzer
            run_btc_analyzer()
        elif role == "reporter":
            from src.services.telegram_notifier import run_daily_reporter
            run_daily_reporter()
        else:
            print(f"Unknown role: {role}", file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        if role == "coin_trader":
            from src.services.telegram_notifier import notify_stop
            notify_stop(symbol)
        print(f"\n[main] {role} stopped by user.")


if __name__ == "__main__":
    main()