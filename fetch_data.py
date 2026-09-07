# fetch_data.py — replace fetch_ohlcv with this

import ccxt
import pandas as pd
import time

def fetch_ohlcv(symbol='ZEC/USDT', timeframe='5m', limit=1000):
    exchange  = ccxt.binance()
    all_ohlcv = []
    since     = None

    # Calculate how many batches we need (binance max 1000 per request)
    batches = (limit // 1000) + 1

    for i in range(batches):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break

            all_ohlcv = ohlcv + all_ohlcv   # prepend older data
            since     = ohlcv[0][0] - (1000 * _timeframe_to_ms(timeframe))
            time.sleep(0.5)                  # avoid rate limit

        except Exception as e:
            print(f"Batch {i} fetch error: {e}")
            break

    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return df.tail(limit)  # return exactly how many you asked for


def _timeframe_to_ms(timeframe):
    units = {'m': 60, 'h': 3600, 'd': 86400}
    return int(timeframe[:-1]) * units[timeframe[-1]] * 1000


def fetch_current_price(symbol='ZEC/USDT'):
    exchange = ccxt.binance()
    ticker   = exchange.fetch_ticker(symbol)
    return ticker['last']