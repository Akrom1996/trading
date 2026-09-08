import ta
import pandas as pd

# Label horizon now matches the barrier model's 2-day window instead of
# the old 15-minute (3-candle) target, so the direction signal and the
# magnitude/probability model are answering the same question.
LABEL_HORIZON_CANDLES = 144   # 2 days of 5-min candles
LABEL_THRESHOLD       = 0.015      # +/-3% over the horizon counts as directional


def add_features(df):
    """
    Computes indicator features only. Safe to call on both training AND
    live data -- does not look into the future, so it never needs to
    drop trailing rows. Only drops leading rows where rolling-window
    indicators (EMA-50, etc.) aren't filled in yet.
    """
    # Trend
    df['ema_9']    = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21']   = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_50']   = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_cross'] = (df['ema_9'] - df['ema_21']) / df['close']  # normalized cross

    # Momentum
    df['rsi']      = ta.momentum.rsi(df['close'], window=14)
    df['rsi_diff'] = df['rsi'].diff()                              # rsi direction
    df['macd']     = ta.trend.macd_diff(df['close'])
    df['stoch']    = ta.momentum.stoch(df['high'], df['low'], df['close'])

    # Volatility
    df['bb_high']  = ta.volatility.bollinger_hband(df['close'])
    df['bb_low']   = ta.volatility.bollinger_lband(df['close'])
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']  # normalized
    df['atr']      = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'])

    # Volume
    df['vwap']     = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['obv']      = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['vol_ma']   = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']                  # volume spike detector

    # Price action
    df['candle_body'] = abs(df['close'] - df['open']) / df['close']
    df['price_pos']   = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)

    # Only drop rows where an INDICATOR is NaN (i.e. leading rows before
    # rolling windows fill up). Does not touch target/label -- those
    # don't exist yet at this stage.
    feature_cols = [
        'ema_9', 'ema_21', 'ema_50', 'ema_cross',
        'rsi', 'rsi_diff', 'macd', 'stoch',
        'bb_high', 'bb_low', 'bb_width', 'atr',
        'vwap', 'obv', 'vol_ma', 'vol_ratio',
        'candle_body', 'price_pos',
    ]
    return df.dropna(subset=feature_cols).reset_index(drop=True)


def add_labels(df, horizon=LABEL_HORIZON_CANDLES, threshold=LABEL_THRESHOLD):
    """
    TRAINING-ONLY. Adds forward-looking 'target'/'label' columns and
    drops the trailing rows that don't have a full forward window yet.

    Do NOT call this on live inference data -- it will drop your most
    recent candles, which is exactly the bug this refactor avoids.

    horizon: how many candles ahead to measure the move (default: 2 days
             of 5-min candles, matching the barrier model's window)
    threshold: +/- move over that horizon required to count as a
               directional label (default 3%) rather than neutral
    """
    df = df.copy()
    df['target'] = (df['close'].shift(-horizon) - df['close']) / df['close']
    df['label'] = df['target'].apply(
        lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
    )
    return df.dropna(subset=['target']).reset_index(drop=True)