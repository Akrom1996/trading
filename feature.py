import ta
import numpy as np
import pandas as pd

# Fibonacci percentage targets up to 13%
FIBONACCI_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.13]

FIB_TIER_NAMES = {
    0: "NEUTRAL / CHOP (<1%)",
    1: "FIB_1 (+1%)",
    2: "FIB_2 (+2%)",
    3: "FIB_3 (+3%)",
    4: "FIB_5 (+5%)",
    5: "FIB_8 (+8%)",
    6: "FIB_13 (+13%)",
}

# Label horizon: 144 5-min candles (12 hours forward).
# A 12-hour window gives the coin ample time to fulfill a 1% to 13% expansion,
# while tracking forward peaks ensures sudden 15-minute bursts are properly credited.
LABEL_HORIZON_CANDLES = 144
DEFAULT_SL_PCT        = 0.02  # 2.0% stop-loss barrier for forward labeling


def add_features(df):
    """
    Computes indicator features only. Safe to call on both training AND
    live data -- does not look into the future, so it never needs to
    drop trailing rows. Only drops leading rows where rolling-window
    indicators (EMA-50, etc.) aren't filled in yet.
    """
    # Trend
    df['ema_9']     = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21']    = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_50']    = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_cross'] = (df['ema_9'] - df['ema_21']) / df['close']  # normalized cross

    # Momentum
    df['rsi']       = ta.momentum.rsi(df['close'], window=14)
    df['rsi_diff']  = df['rsi'].diff()                             # rsi direction
    df['macd']      = ta.trend.macd_diff(df['close'])
    df['stoch']     = ta.momentum.stoch(df['high'], df['low'], df['close'])

    # Volatility
    df['bb_high']   = ta.volatility.bollinger_hband(df['close'])
    df['bb_low']    = ta.volatility.bollinger_lband(df['close'])
    df['bb_width']  = (df['bb_high'] - df['bb_low']) / df['close']  # normalized
    df['atr']       = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'])
    df['atr_ratio'] = df['atr'] / df['close']

    # Volume
    df['vwap']          = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['obv']           = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['vol_ma']        = df['volume'].rolling(20).mean()
    df['vol_ratio']     = df['volume'] / (df['vol_ma'] + 1e-9)       # volume spike detector
    df['vol_ratio_15m'] = df['volume'].rolling(3).mean() / (df['vol_ma'] + 1e-9)

    # Price action & short-term momentum (detects rapid 5m / 15m / 30m / 1h surges)
    df['candle_body'] = abs(df['close'] - df['open']) / df['close']
    df['price_pos']   = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    df['return_5m']   = df['close'].pct_change(1)
    df['return_15m']  = df['close'].pct_change(3)   # 15-minute jump detector
    df['return_30m']  = df['close'].pct_change(6)   # 30-minute trend
    df['return_1h']   = df['close'].pct_change(12)  # 1-hour trend
    df['range_15m']   = (df['high'].rolling(3).max() - df['low'].rolling(3).min()) / df['close']

    # Only drop rows where an INDICATOR is NaN (i.e. leading rows before
    # rolling windows fill up). Does not touch target/label -- those
    # don't exist yet at this stage.
    feature_cols = [
        'ema_9', 'ema_21', 'ema_50', 'ema_cross',
        'rsi', 'rsi_diff', 'macd', 'stoch',
        'bb_high', 'bb_low', 'bb_width', 'atr', 'atr_ratio',
        'vwap', 'obv', 'vol_ma', 'vol_ratio', 'vol_ratio_15m',
        'candle_body', 'price_pos',
        'return_5m', 'return_15m', 'return_30m', 'return_1h',
        'range_15m',
    ]
    return df.dropna(subset=feature_cols).reset_index(drop=True)


def add_labels(df, horizon=LABEL_HORIZON_CANDLES, sl_pct=DEFAULT_SL_PCT):
    """
    TRAINING-ONLY. Evaluates the maximum forward upward move across
    Fibonacci targets (1%, 2%, 3%, 5%, 8%, 13%) before stop loss is hit.

    Labels:
      0: No profitable Fibonacci increase (hit SL first, or max gain < 1%)
      1: Reached +1% Fib tier before SL
      2: Reached +2% Fib tier before SL
      3: Reached +3% Fib tier before SL
      4: Reached +5% Fib tier before SL
      5: Reached +8% Fib tier before SL
      6: Reached +13% Fib tier before SL

    Drops the trailing rows that don't have a full forward window yet.
    """
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    n      = len(df)

    labels = np.zeros(n, dtype=int)
    max_gains = np.zeros(n, dtype=float)

    for i in range(n - horizon):
        entry    = closes[i]
        sl_price = entry * (1.0 - sl_pct)

        # Track forward peak before stop-loss is breached
        peak_gain = 0.0
        for j in range(i + 1, i + 1 + horizon):
            if lows[j] <= sl_price:
                # Stopped out -- stop scanning forward
                break
            gain = (highs[j] - entry) / entry
            if gain > peak_gain:
                peak_gain = gain

        max_gains[i] = peak_gain

        # Determine Fibonacci tier achieved
        tier = 0
        if peak_gain >= 0.13:
            tier = 6
        elif peak_gain >= 0.08:
            tier = 5
        elif peak_gain >= 0.05:
            tier = 4
        elif peak_gain >= 0.03:
            tier = 3
        elif peak_gain >= 0.02:
            tier = 2
        elif peak_gain >= 0.01:
            tier = 1

        labels[i] = tier

    out = df.iloc[:n - horizon].copy()
    out['max_gain'] = max_gains[:n - horizon]
    out['label']    = labels[:n - horizon]
    return out.reset_index(drop=True)