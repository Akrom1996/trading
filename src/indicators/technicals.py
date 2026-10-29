"""
Technical indicators, ML feature engineering, Fibonacci barriers, model training, and signal generation.
Purely consolidated in src/indicators/technicals.py.
"""

from __future__ import annotations
import math
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    import numpy as np
    import pandas as pd
    import ta
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from xgboost import XGBClassifier
    HAS_ML_DEPS = True
except ImportError:
    np = None
    pd = None
    ta = None
    HAS_ML_DEPS = False

from config.settings import settings

# ── Fibonacci Tiers & Horizons ─────────────────────────────────
FIBONACCI_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.08]

FIB_TIER_NAMES = {
    0: "NEUTRAL / CHOP (<1%)",
    1: "FIB_1 (+1%)",
    2: "FIB_2 (+2%)",
    3: "FIB_3 (+3%)",
    4: "FIB_5 (+5%)",
    5: "FIB_8 (+8%+)",
}

TIER_TO_PCT = {
    0: 0.0,
    1: 0.01,
    2: 0.02,
    3: 0.03,
    4: 0.05,
    5: 0.08,
}

LABEL_HORIZON_CANDLES = 144
DEFAULT_SL_PCT = getattr(settings, 'MAX_RISK_PER_TRADE', 0.03)
DEFAULT_TP_PCT = 0.03
TP_PCT = DEFAULT_TP_PCT
SL_PCT = DEFAULT_SL_PCT

TIER_CHOP = 0

LABEL_SL_FIRST = 0
LABEL_FIB_1 = 1
LABEL_FIB_2 = 2
LABEL_FIB_3 = 3
LABEL_FIB_5 = 4
LABEL_FIB_8 = 5
LABEL_TIMEOUT = 7  # kept at 7 on purpose: old models with class 6 (former FIB_13) fail the label guard and retrain
LABEL_TP_FIRST = LABEL_FIB_2

LABEL_NAMES = {
    LABEL_SL_FIRST: f"SL_FIRST (tier-1 -{FIBONACCI_LEVELS[0]*100:.0f}%)",
    LABEL_FIB_1: "TP_FIB_1 (+1%)",
    LABEL_FIB_2: "TP_FIB_2 (+2%)",
    LABEL_FIB_3: "TP_FIB_3 (+3%)",
    LABEL_FIB_5: "TP_FIB_5 (+5%)",
    LABEL_FIB_8: "TP_FIB_8 (+8%+)",
    LABEL_TIMEOUT: "TIMEOUT",
}

MIN_HEALTHY_CLASS_SHARE = 0.03

FEATURE_COLUMNS = [
    'rsi', 'rsi_diff', 'macd', 'stoch',
    'ema_9', 'ema_21', 'ema_50', 'ema_cross',
    'atr', 'atr_ratio', 'bb_high', 'bb_low', 'bb_width',
    'obv', 'vwap', 'vol_ratio', 'vol_ratio_15m',
    'candle_body', 'price_pos', 'volume',
    'return_5m', 'return_15m', 'return_30m', 'return_1h',
    'range_15m',
]
FEATURES = FEATURE_COLUMNS

FLOW_WINDOW = 288  # candles for rolling VWAP / OBV (1 day of 5m); must stay well below LIVE_CANDLES

# Bump when a feature's definition changes without changing the column count,
# so models saved with the old definition are retrained instead of reused.
FEATURE_VERSION = 3  # v2: rolling VWAP / OBV instead of cumulative; v3: per-tier stops in labels

OOS_TEST_FRACTION = 0.2
WIN_COLUMNS = [f'win_tier_{k}' for k in range(1, len(FIBONACCI_LEVELS) + 1)]


def _labels_are_valid(encoder: Any) -> bool:
    """True if every class the saved encoder knows still exists in the current label set."""
    valid_labels = set(LABEL_NAMES) | set(FIB_TIER_NAMES)
    return all(int(c) in valid_labels for c in encoder.classes_)


# ── Technical Indicators & Features ─────────────────────────────

def detect_rsi_divergence(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Identifies regular bullish and bearish RSI divergences."""
    div = pd.Series(0, index=df.index)
    if len(df) < window * 2:
        return div

    price_low = df['low'].rolling(window=window).min()
    price_high = df['high'].rolling(window=window).max()
    rsi_low = df['rsi'].rolling(window=window).min()
    rsi_high = df['rsi'].rolling(window=window).max()

    prev_price_low = price_low.shift(window)
    prev_price_high = price_high.shift(window)
    prev_rsi_low = rsi_low.shift(window)
    prev_rsi_high = rsi_high.shift(window)

    bullish = (df['low'] <= prev_price_low) & (df['rsi'] > prev_rsi_low) & (df['rsi'] < 45)
    bearish = (df['high'] >= prev_price_high) & (df['rsi'] < prev_rsi_high) & (df['rsi'] > 55)

    div.loc[bullish] = 1
    div.loc[bearish] = -1
    return div


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicator features for live inference or training."""
    if not HAS_ML_DEPS:
        raise ImportError("pandas, numpy, and ta are required for feature calculation.")

    df = df.copy()

    # EMAs & Cross
    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_cross'] = (df['ema_9'] - df['ema_21']) / df['close']

    # Momentum & RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['rsi_diff'] = df['rsi'].diff()
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['rsi_div'] = detect_rsi_divergence(df)

    # Volatility
    df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
    df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['atr_ratio'] = df['atr'] / df['close']

    # Volume & Flow
    # Rolling (not cumulative) so values don't depend on how many candles were fetched:
    # training uses TRAIN_CANDLES, live inference uses LIVE_CANDLES.
    pv_sum = (df['close'] * df['volume']).rolling(FLOW_WINDOW).sum()
    df['vwap'] = pv_sum / (df['volume'].rolling(FLOW_WINDOW).sum() + 1e-9)
    signed_volume = np.sign(df['close'].diff()).fillna(0) * df['volume']
    df['obv'] = signed_volume.rolling(FLOW_WINDOW).sum()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-9)
    df['vol_ratio_15m'] = df['volume'].rolling(3).mean() / (df['vol_ma'] + 1e-9)

    # Price Action & Returns
    df['candle_body'] = abs(df['close'] - df['open']) / df['close']
    df['price_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    df['return_5m'] = df['close'].pct_change(1)
    df['return_15m'] = df['close'].pct_change(3)
    df['return_30m'] = df['close'].pct_change(6)
    df['return_1h'] = df['close'].pct_change(12)
    df['range_15m'] = (df['high'].rolling(3).max() - df['low'].rolling(3).min()) / df['close']

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


def _forward_peak_gains(df: pd.DataFrame, horizon: int, sl_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each candle i in [0, n - horizon), looks at candles i+1..i+horizon and returns
    (peak gain of the highs before the first candle whose low hits SL, whether SL was hit).
    Vectorized equivalent of the per-candle forward scan.
    """
    closes = df['close'].to_numpy(dtype=float)
    highs = df['high'].to_numpy(dtype=float)
    lows = df['low'].to_numpy(dtype=float)
    m = len(df) - horizon
    if m <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=bool)

    windows = np.lib.stride_tricks.sliding_window_view
    fwd_highs = windows(highs[1:], horizon)[:m]
    fwd_lows = windows(lows[1:], horizon)[:m]
    entry = closes[:m]

    hit = fwd_lows <= (entry * (1.0 - sl_pct))[:, None]
    stopped_out = hit.any(axis=1)
    first_hit = np.where(stopped_out, hit.argmax(axis=1), horizon)
    before_sl = np.arange(horizon)[None, :] < first_hit[:, None]

    peak_high = np.where(before_sl, fwd_highs, -np.inf).max(axis=1)
    peak_gain = np.maximum((peak_high - entry) / entry, 0.0)
    return peak_gain, stopped_out


def tier_stop_pct(level: float) -> float:
    """Live stop for a tier: SL = min(TP, MAX_RISK_PER_TRADE)."""
    return min(level, getattr(settings, 'MAX_RISK_PER_TRADE', DEFAULT_SL_PCT))


def _tier_outcomes(df: pd.DataFrame, horizon: int = LABEL_HORIZON_CANDLES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-tier trade outcomes matching live execution: tier k (level L) wins if the high
    reaches entry*(1+L) before the low hits entry*(1-min(L, cap)). A candle that crosses
    both counts as a loss (_forward_peak_gains only uses highs before the stop candle).
    Returns (wins[n, len(FIBONACCI_LEVELS)], tier-1 trade stopped out[n]).
    """
    gains_by_sl: Dict[float, np.ndarray] = {}
    stopped_by_sl: Dict[float, np.ndarray] = {}
    for level in FIBONACCI_LEVELS:
        sl = tier_stop_pct(level)
        if sl not in gains_by_sl:
            gains_by_sl[sl], stopped_by_sl[sl] = _forward_peak_gains(df, horizon, sl)

    wins = np.column_stack([
        gains_by_sl[tier_stop_pct(level)] >= level for level in FIBONACCI_LEVELS
    ]) if len(df) > horizon else np.zeros((0, len(FIBONACCI_LEVELS)), dtype=bool)
    first_sl = tier_stop_pct(FIBONACCI_LEVELS[0])
    stopped_tier1 = stopped_by_sl[first_sl] & ~wins[:, 0] if len(wins) else np.zeros(0, dtype=bool)
    return wins, stopped_tier1


def _highest_winning_tier(wins: np.ndarray) -> np.ndarray:
    """1-based index of the highest winning tier per row, 0 if none won."""
    n_tiers = wins.shape[1]
    return np.where(wins.any(axis=1), n_tiers - np.argmax(wins[:, ::-1], axis=1), 0)


def add_labels(df: pd.DataFrame, horizon: int = LABEL_HORIZON_CANDLES) -> pd.DataFrame:
    """Labels the highest Fibonacci tier won with its live stop (min(TP, cap)), 0 if none."""
    peak_gain, _ = _forward_peak_gains(df, horizon, getattr(settings, 'MAX_RISK_PER_TRADE', DEFAULT_SL_PCT))
    wins, _ = _tier_outcomes(df, horizon)

    # Tiers 0..5 for none, 1%, 2%, 3%, 5%, 8%
    labels = _highest_winning_tier(wins)

    out = df.iloc[:len(labels)].copy()
    out['max_gain'] = peak_gain
    out['label'] = labels.astype(int)
    out[WIN_COLUMNS] = wins
    return out.reset_index(drop=True)


def label_triple_barrier(df: pd.DataFrame, horizon: int = LABEL_HORIZON_CANDLES) -> pd.DataFrame:
    """Triple-barrier labeling for spot trades, one live stop per tier."""
    wins, stopped_tier1 = _tier_outcomes(df, horizon)

    # FIB tiers share numbering with add_labels (1..5); no win splits into SL_FIRST vs TIMEOUT
    tiers = _highest_winning_tier(wins)
    labels = np.where(tiers > 0, tiers, np.where(stopped_tier1, LABEL_SL_FIRST, LABEL_TIMEOUT))

    out = df.iloc[:len(labels)].copy()
    out['barrier_label'] = labels.astype(int)
    out[WIN_COLUMNS] = wins
    return out


def _purged_split(n: int, horizon: int = LABEL_HORIZON_CANDLES) -> Tuple[np.ndarray, np.ndarray]:
    """Time-ordered split: last OOS_TEST_FRACTION is test; the horizon rows before it are purged."""
    test_start = int(n * (1 - OOS_TEST_FRACTION))
    train_end = max(test_start - horizon, 0)
    return np.arange(train_end), np.arange(test_start, n)


def _make_classifier(num_classes: int, n_jobs: int = -1, **extra) -> Any:
    is_multiclass = num_classes > 2
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective='multi:softprob' if is_multiclass else 'binary:logistic',
        num_class=num_classes if is_multiclass else None,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=n_jobs,
        **extra,
    )


def _fit_holdout(X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray, **extra):
    """Fits scaler + encoder + model on train rows only; returns (test probabilities, encoder) or None."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    y_train = y.iloc[train_idx]
    if y_train.nunique() < 2:
        return None
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_train)
    scaler = StandardScaler()
    model = _make_classifier(len(encoder.classes_), **extra)
    model.fit(scaler.fit_transform(X.iloc[train_idx]), y_enc)
    probs = model.predict_proba(scaler.transform(X.iloc[test_idx]))
    return probs, encoder


# ── Trade simulation (shared by the hourly OOS and the walk-forward) ──

def simulate_trades(ohlc: pd.DataFrame, signal_mask: np.ndarray, tier_idx: np.ndarray,
                    start: int, end: int, horizon: int = LABEL_HORIZON_CANDLES) -> List[Dict[str, Any]]:
    """
    Replays entries the way the bot trades them, one position at a time:
      - signal on closed candle i -> limit at close*(1-ENTRY_OFFSET_PCT), filled if a later candle's low
        reaches it within PENDING_FILL_TIMEOUT_MINUTES, otherwise cancelled
      - TP/SL from the fill at the tier's level and live stop min(L, cap); on the fill candle only the SL
        counts, afterwards a candle touching both counts as SL; no hit within `horizon` -> exit at close
      - the next signal is only considered after the trade exits (or the unfilled order times out)
    Signals are taken from [start, end); trades may resolve past `end`. Trades the data can't resolve
    are marked 'incomplete'. tier_idx holds 1-based tiers per row.
    """
    highs = ohlc['high'].to_numpy(dtype=float)
    lows = ohlc['low'].to_numpy(dtype=float)
    closes = ohlc['close'].to_numpy(dtype=float)
    n = len(ohlc)
    fill_candles = max(1, math.ceil(settings.PENDING_FILL_TIMEOUT_MINUTES / 5))
    offset = settings.ENTRY_OFFSET_PCT
    fee = getattr(settings, 'FEE_ROUND_TRIP', 0.0)

    trades: List[Dict[str, Any]] = []
    i = start
    while i < min(end, n):
        if not signal_mask[i]:
            i += 1
            continue
        tier = int(tier_idx[i])
        level = FIBONACCI_LEVELS[tier - 1]
        sl = tier_stop_pct(level)
        limit = closes[i] * (1 - offset)

        fill_j = next((j for j in range(i + 1, min(i + 1 + fill_candles, n)) if lows[j] <= limit), None)
        if fill_j is None:
            outcome = 'unfilled' if i + fill_candles < n else 'incomplete'
            trades.append({'signal_idx': i, 'tier': tier, 'filled': False, 'outcome': outcome})
            i += fill_candles + 1
            continue

        tp_price, sl_price = limit * (1 + level), limit * (1 - sl)
        exit_idx, gross, outcome = None, None, None
        if lows[fill_j] <= sl_price:
            exit_idx, gross, outcome = fill_j, -sl, 'sl'
        else:
            for k in range(fill_j + 1, min(fill_j + horizon, n - 1) + 1):
                if lows[k] <= sl_price:
                    exit_idx, gross, outcome = k, -sl, 'sl'
                    break
                if highs[k] >= tp_price:
                    exit_idx, gross, outcome = k, level, 'tp'
                    break
            if exit_idx is None and fill_j + horizon <= n - 1:
                exit_idx = fill_j + horizon
                gross, outcome = closes[exit_idx] / limit - 1, 'timeout'

        if exit_idx is None:
            trades.append({'signal_idx': i, 'tier': tier, 'filled': True, 'outcome': 'incomplete'})
            break
        trades.append({
            'signal_idx': i, 'fill_idx': fill_j, 'exit_idx': exit_idx, 'tier': tier, 'filled': True,
            'outcome': outcome, 'net': gross - fee,
        })
        i = exit_idx + 1
    return trades


def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stats over resolved trades: count, fill rate, win rate, avg/total net, max drawdown (all fractions)."""
    signals = [t for t in trades if t['outcome'] != 'incomplete']
    done = [t for t in signals if t['filled']]
    nets = np.array([t['net'] for t in done], dtype=float)
    if len(nets):
        equity = np.cumsum(nets)
        max_dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], equity])) - np.concatenate([[0.0], equity])))
    else:
        max_dd = 0.0
    return {
        'trades': int(len(nets)),
        'signals': int(len(signals)),
        'fill_rate': round(len(done) / len(signals), 4) if signals else None,
        'win_rate': round(float((nets > 0).mean()), 4) if len(nets) else None,
        'avg_net': round(float(nets.mean()), 5) if len(nets) else None,
        'total_net': round(float(nets.sum()), 5),
        'max_drawdown': round(max_dd, 5),
    }


def median_tier(trades: List[Dict[str, Any]], default: int = 2) -> int:
    tiers = [t['tier'] for t in trades if t.get('filled')]
    return int(round(float(np.median(tiers)))) if tiers else default


def directional_buy_mask(model: Any, scaler: Any, encoder: Any, X: pd.DataFrame,
                         base_rate_bullish: Optional[float], min_confidence: float = 0.40) -> np.ndarray:
    """Vectorized generate_signal(): True where the directional model would say BUY."""
    proba = model.predict_proba(scaler.transform(X))
    classes = np.asarray(encoder.classes_).astype(int)
    pred = classes[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)
    p_bullish = proba[:, classes >= 1].sum(axis=1)
    return np.array([_is_buy(int(pr), float(c), float(pb), base_rate_bullish, min_confidence)
                     for pr, c, pb in zip(pred, confidence, p_bullish)], dtype=bool)


def _is_buy(pred: int, confidence: float, p_bullish: float, base_rate_bullish: Optional[float],
            min_confidence: float) -> bool:
    # Bullish shortcut only counts if P(bullish) clears the training base rate by SIGNAL_MARGIN
    bullish_edge = (base_rate_bullish is not None
                    and p_bullish >= base_rate_bullish + getattr(settings, 'SIGNAL_MARGIN', 0.0))
    return pred >= 1 and (confidence >= min_confidence or bullish_edge)


# ── Directional Model (train, save, load) ───────────────────────

def _get_model_paths(symbol: str) -> Dict[str, str]:
    safe = symbol.replace('/', '').upper()
    state_dir = settings.STATE_DIR
    if state_dir and not os.path.exists(state_dir):
        try:
            os.makedirs(state_dir, exist_ok=True)
        except Exception:
            pass

    base_dir = state_dir if os.path.exists(state_dir) else "."
    return {
        'model': os.path.join(base_dir, f'saved_model_{safe}.pkl'),
        'scaler': os.path.join(base_dir, f'saved_scaler_{safe}.pkl'),
        'encoder': os.path.join(base_dir, f'saved_encoder_{safe}.pkl'),
        'metadata': os.path.join(base_dir, f'model_metadata_{safe}.pkl'),
    }


def _directional_oos(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Purged holdout stats for the directional model (logging only)."""
    train_idx, test_idx = _purged_split(len(df))
    fitted = _fit_holdout(X, y, train_idx, test_idx, subsample=0.8, colsample_bytree=0.8)
    if fitted is None:
        print("  OOS directional: not enough data for a holdout")
        return {'test_rows': int(len(test_idx))}
    probs, encoder = fitted
    preds = encoder.classes_[probs.argmax(axis=1)].astype(int)
    y_test = y.iloc[test_idx].to_numpy(dtype=int)
    accuracy = float((preds == y_test).mean())

    # A predicted tier k >= 1 is scored as the live trade at tier k: did it win?
    buy_rows = preds >= 1
    tier_wins = df.iloc[test_idx][WIN_COLUMNS].to_numpy(dtype=bool)
    buy_wins = tier_wins[np.flatnonzero(buy_rows), preds[buy_rows] - 1]
    buy_win_rate = float(buy_wins.mean()) if buy_rows.any() else None
    stats = {
        'test_rows': int(len(test_idx)),
        'accuracy': round(accuracy, 4),
        'buy_rate': round(float(buy_rows.mean()), 4),
        'buy_win_rate': round(buy_win_rate, 4) if buy_win_rate is not None else None,
    }
    win_str = f"{buy_win_rate:.0%}" if buy_win_rate is not None else "n/a"
    print(f"  OOS directional: test={stats['test_rows']} acc={accuracy:.1%} "
          f"buy_rate={stats['buy_rate']:.1%} buy_win={win_str}")
    return stats


def train_model(df: pd.DataFrame, oos: bool = True, verbose: bool = True,
                n_jobs: int = -1) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    cap = getattr(settings, 'MAX_RISK_PER_TRADE', DEFAULT_SL_PCT)
    if verbose:
        print(f"  Label distribution (Fibonacci Tiers, stop per tier = min(TP, {cap*100:.1f}%)):")
    total = len(df)
    class_freqs: Dict[int, float] = {}
    for tier, name in FIB_TIER_NAMES.items():
        count = int((df['label'] == tier).sum())
        pct = (count / total * 100) if total else 0.0
        class_freqs[tier] = round(count / total, 4) if total else 0.0
        if count > 0 and verbose:
            print(f"    Tier {tier} ({name:22s}): {count:5d} ({pct:.1f}%)")

    X = df[FEATURES]
    y = df['label']
    base_rate_bullish = float((y >= 1).mean()) if total else 0.0
    if verbose:
        print(f"  Base rate P(bullish tier) = {base_rate_bullish:.1%}")

    oos_stats = _directional_oos(df, X, y) if oos else None

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # No class weighting: probabilities must stay on the real-frequency scale of base_rate_bullish
    model = _make_classifier(len(encoder.classes_), n_jobs=n_jobs, subsample=0.8, colsample_bytree=0.8)
    model.fit(X_scaled, y_encoded)
    train_meta = {
        'class_freqs': class_freqs,
        'base_rate_bullish': round(base_rate_bullish, 4),
        'oos': oos_stats,
    }
    return model, scaler, encoder, train_meta


def save_model(model: Any, scaler: Any, encoder: Any, symbol: str, candles_count: Optional[int] = None,
               extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    paths = _get_model_paths(symbol)
    with open(paths['model'], 'wb') as f:
        pickle.dump(model, f)
    with open(paths['scaler'], 'wb') as f:
        pickle.dump(scaler, f)
    with open(paths['encoder'], 'wb') as f:
        pickle.dump(encoder, f)
    metadata = {
        'trained_at': datetime.now(),
        'candles_count': candles_count,
        'symbol': symbol,
        'feature_version': FEATURE_VERSION,
        **(extra_meta or {}),
    }
    with open(paths['metadata'], 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Model saved to {paths['model']}")
    return metadata


def load_model(symbol: str) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    paths = _get_model_paths(symbol)
    # Check fallback in current directory if not found in state_dir
    safe = symbol.replace('/', '').upper()
    if not os.path.exists(paths['model']) and os.path.exists(f'saved_model_{safe}.pkl'):
        paths = {
            'model': f'saved_model_{safe}.pkl',
            'scaler': f'saved_scaler_{safe}.pkl',
            'encoder': f'saved_encoder_{safe}.pkl',
            'metadata': f'model_metadata_{safe}.pkl',
        }

    if not all(os.path.exists(p) for p in paths.values()):
        return None, None, None, None

    try:
        with open(paths['model'], 'rb') as f:
            model = pickle.load(f)
        with open(paths['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        with open(paths['encoder'], 'rb') as f:
            encoder = pickle.load(f)
        with open(paths['metadata'], 'rb') as f:
            metadata = pickle.load(f)

        if getattr(scaler, 'n_features_in_', None) != len(FEATURES):
            print(f"  Feature set changed ({getattr(scaler, 'n_features_in_', '?')} -> {len(FEATURES)}), retraining...")
            return None, None, None, None

        if not _labels_are_valid(encoder):
            print("  Label set changed, retraining...")
            return None, None, None, None

        if metadata.get('feature_version') != FEATURE_VERSION:
            print(f"  Feature definitions changed (v{metadata.get('feature_version', 1)} -> v{FEATURE_VERSION}), retraining...")
            return None, None, None, None

        if metadata.get('base_rate_bullish') is None:
            print("  Model metadata has no class base rates, retraining...")
            return None, None, None, None

        return model, scaler, encoder, metadata
    except Exception as e:
        print(f"  Load model error: {e}")
        return None, None, None, None


def model_is_fresh(metadata: Optional[Dict[str, Any]], max_age_hours: int = 12) -> bool:
    if metadata is None or 'trained_at' not in metadata:
        return False
    age_hours = (datetime.now() - metadata['trained_at']).total_seconds() / 3600
    return age_hours < max_age_hours


# ── Barrier Model (train, save, load, predict, enter) ───────────

def _get_barrier_path(symbol: str) -> str:
    safe = symbol.replace('/', '').upper()
    state_dir = settings.STATE_DIR
    if state_dir and not os.path.exists(state_dir):
        try:
            os.makedirs(state_dir, exist_ok=True)
        except Exception:
            pass
    base_dir = state_dir if os.path.exists(state_dir) else "."
    return os.path.join(base_dir, f"barrier_model_{safe}.pkl")


def ev_entry_rule(full_probs: Dict[int, float]) -> Dict[str, Any]:
    """
    Expected value per tier k (level L, live stop S = min(L, cap)):
        p_win_k = P(label >= k);  EV_k = p_win_k*L - (1 - p_win_k)*S - FEE_ROUND_TRIP
    Picks the max-EV tier; enter only if EV >= MIN_EDGE_PCT.
    """
    fee = getattr(settings, 'FEE_ROUND_TRIP', 0.0)
    min_edge = getattr(settings, 'MIN_EDGE_PCT', 0.0)
    tier_labels = [LABEL_FIB_1, LABEL_FIB_2, LABEL_FIB_3, LABEL_FIB_5, LABEL_FIB_8]

    ev_by_tier = []
    for k, level in enumerate(FIBONACCI_LEVELS, start=1):
        p_win = sum(full_probs.get(lbl, 0.0) for lbl in tier_labels[k - 1:])
        sl = tier_stop_pct(level)
        ev = p_win * level - (1 - p_win) * sl - fee
        ev_by_tier.append({'tier': k, 'tp': level, 'sl': sl, 'p_win': p_win, 'ev': ev})

    best = max(ev_by_tier, key=lambda t: t['ev'])
    return {
        'chosen_tier': best['tier'],
        'ev': round(best['ev'], 5),
        'p_win': round(best['p_win'], 4),
        'recommended_tp': best['tp'],
        'recommended_sl': round(best['sl'], 4),
        'enter': best['ev'] >= min_edge,
        'ev_by_tier': [{**t, 'p_win': round(t['p_win'], 4), 'ev': round(t['ev'], 5)} for t in ev_by_tier],
    }


def _barrier_oos(labeled: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Purged holdout (diagnostic only): EV-rule entries on the test segment replayed as non-overlapping
    trades with the bot's fill rules, against a baseline that enters on every flat candle at the same tier.
    """
    train_idx, test_idx = _purged_split(len(labeled))
    fitted = _fit_holdout(X, y, train_idx, test_idx)
    if fitted is None:
        print("  OOS barrier: not enough data for a holdout")
        return {'test_rows': int(len(test_idx)), 'trades': 0}
    probs, encoder = fitted

    n = len(labeled)
    mask = np.zeros(n, dtype=bool)
    tiers = np.ones(n, dtype=int)
    for row, row_probs in zip(test_idx, probs):
        rule = ev_entry_rule({int(c): float(p) for c, p in zip(encoder.classes_, row_probs)})
        if rule['enter']:
            mask[row] = True
            tiers[row] = rule['chosen_tier']

    test_start = int(test_idx[0])
    model_trades = simulate_trades(labeled, mask, tiers, test_start, n)
    base_tier = median_tier(model_trades)
    baseline_trades = simulate_trades(labeled, np.ones(n, dtype=bool), np.full(n, base_tier), test_start, n)

    stats = summarize_trades(model_trades)
    base = summarize_trades(baseline_trades)
    edge = (stats['avg_net'] - base['avg_net']
            if stats['avg_net'] is not None and base['avg_net'] is not None else None)
    stats.update({
        'test_rows': int(len(test_idx)),
        'baseline_trades': base['trades'],
        'baseline_avg_net': base['avg_net'],
        'baseline_tier': base_tier,
        'edge': round(edge, 5) if edge is not None else None,
    })

    def pct(v):
        return f"{v*100:+.2f}%" if v is not None else "n/a"
    win = f"{stats['win_rate']:.0%}" if stats['win_rate'] is not None else "n/a"
    print(f"  OOS barrier: trades={stats['trades']} (non-overlapping) win={win} avg={pct(stats['avg_net'])} "
          f"| baseline avg={pct(base['avg_net'])} | edge={pct(edge)}")
    return stats


def train_barrier_model(df: pd.DataFrame, symbol: Optional[str] = None, oos: bool = True,
                        verbose: bool = True, n_jobs: int = -1) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    labeled = label_triple_barrier(df)
    total = len(labeled)
    label_stats = {}
    cap = getattr(settings, 'MAX_RISK_PER_TRADE', DEFAULT_SL_PCT)
    if verbose:
        print(f"  Label distribution{f' ({symbol})' if symbol else ''} (stop per tier = min(TP, {cap*100:.1f}%)):")
    for lbl, name in LABEL_NAMES.items():
        count = int((labeled['barrier_label'] == lbl).sum())
        pct = count / total * 100 if total else 0.0
        label_stats[name] = {'count': count, 'pct': round(pct, 1)}
        # TIMEOUT needs price to stay within ±1% for the whole horizon, so it's ~always empty; no warning
        low = pct / 100 < MIN_HEALTHY_CLASS_SHARE and lbl != LABEL_TIMEOUT
        if verbose:
            print(f"    {name:20s}: {count:5d} ({pct:.1f}%){' ⚠️ low' if low else ''}")

    X = labeled[FEATURES]
    y = labeled['barrier_label']

    label_stats['oos'] = _barrier_oos(labeled, X, y) if oos else None

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Unweighted so predicted probabilities reflect real frequencies (the EV rule depends on it)
    model = _make_classifier(len(encoder.classes_), n_jobs=n_jobs)
    model.fit(X_scaled, y_encoded)
    return model, scaler, encoder, label_stats


def save_barrier_model(model: Any, scaler: Any, encoder: Any, symbol: str, candles_count: Optional[int] = None, label_stats: Optional[Dict[str, Any]] = None):
    path = _get_barrier_path(symbol)
    with open(path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'encoder': encoder,
            'symbol': symbol,
            'trained_at': datetime.now(),
            'candles_count': candles_count,
            'label_stats': label_stats,
            'oos': (label_stats or {}).get('oos'),
            'feature_version': FEATURE_VERSION,
        }, f)


def load_barrier_model(symbol: str) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    path = _get_barrier_path(symbol)
    safe = symbol.replace('/', '').upper()
    if not os.path.exists(path) and os.path.exists(f"barrier_model_{safe}.pkl"):
        path = f"barrier_model_{safe}.pkl"

    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        scaler = data.get('scaler')
        if getattr(scaler, 'n_features_in_', None) != len(FEATURES):
            print(f"  Feature set changed ({getattr(scaler, 'n_features_in_', '?')} -> {len(FEATURES)}), retraining barrier model...")
            return None, None, None, None
        if not _labels_are_valid(data['encoder']):
            print("  Label set changed, retraining barrier model...")
            return None, None, None, None
        if data.get('feature_version') != FEATURE_VERSION:
            print(f"  Feature definitions changed (v{data.get('feature_version', 1)} -> v{FEATURE_VERSION}), retraining barrier model...")
            return None, None, None, None
        return data['model'], data['scaler'], data['encoder'], data
    except Exception:
        return None, None, None, None


def predict_barrier_probabilities(model: Any, scaler: Any, encoder: Any, df_latest: pd.DataFrame) -> Dict[str, Any]:
    X = df_latest[FEATURES].iloc[[-1]]
    X_scaled = scaler.transform(X)
    raw_probs = model.predict_proba(X_scaled)[0]

    full_probs = {lbl: 0.0 for lbl in LABEL_NAMES}
    for col_idx, original_label in enumerate(encoder.classes_):
        full_probs[original_label] = float(raw_probs[col_idx])

    p_fib_1 = full_probs[LABEL_FIB_1]
    p_fib_2 = full_probs[LABEL_FIB_2]
    p_fib_3 = full_probs[LABEL_FIB_3]
    p_fib_5 = full_probs[LABEL_FIB_5]
    p_fib_8 = full_probs[LABEL_FIB_8]

    p_profitable = p_fib_1 + p_fib_2 + p_fib_3 + p_fib_5 + p_fib_8
    p_extended = p_fib_3 + p_fib_5 + p_fib_8

    rule = ev_entry_rule(full_probs)

    return {
        'p_sl_first': round(full_probs[LABEL_SL_FIRST], 4),
        'p_tp_first': round(p_profitable, 4),
        'p_extended': round(p_extended, 4),
        'p_timeout': round(full_probs[LABEL_TIMEOUT], 4),
        'p_fib_1': round(p_fib_1, 4),
        'p_fib_2': round(p_fib_2, 4),
        'p_fib_3': round(p_fib_3, 4),
        'p_fib_5': round(p_fib_5, 4),
        'p_fib_8': round(p_fib_8, 4),
        **rule,
    }


def should_enter(barrier_probs: Dict[str, Any]) -> Tuple[bool, str]:
    tp = barrier_probs.get('recommended_tp', DEFAULT_TP_PCT)
    sl = barrier_probs.get('recommended_sl', tier_stop_pct(tp))
    p = barrier_probs.get('p_win', 0.0)
    ev = barrier_probs.get('ev', 0.0)
    min_edge = getattr(settings, 'MIN_EDGE_PCT', 0.0)
    desc = (f"Tier {barrier_probs.get('chosen_tier')} +{tp*100:.0f}%/-{sl*100:.1f}%: "
            f"P(win)={p:.0%}, EV={ev*100:+.2f}% net")
    if barrier_probs.get('enter'):
        return True, f"{desc} ≥ {min_edge*100:.2f}% edge — entering"
    return False, f"{desc} below {min_edge*100:.2f}% edge — skipping"


# ── Signal Generation ───────────────────────────────────────────

def generate_signal(model: Any, scaler: Any, encoder: Any, df: pd.DataFrame, min_confidence: float = 0.40,
                    base_rate_bullish: Optional[float] = None) -> Dict[str, Any]:
    latest = df[FEATURES].iloc[-1:].copy()
    scaled = scaler.transform(latest)

    proba = model.predict_proba(scaled)[0]
    pred_enc = int(np.argmax(proba))
    confidence = float(max(proba))

    pred = encoder.inverse_transform([pred_enc])[0]
    class_probs = {cls: prob for cls, prob in zip(encoder.classes_, proba)}
    p_bullish = sum(prob for cls, prob in class_probs.items() if cls >= 1)
    p_chop = class_probs.get(TIER_CHOP, 0.0)

    current_price = float(df['close'].iloc[-1])

    signal = {
        'action': None,
        'confidence': confidence,
        'p_bullish': round(p_bullish, 4),
        'p_chop': round(p_chop, 4),
        'p_sl_first': round(p_chop, 4),
        'pred_label': int(pred),
        'fib_tier': int(pred) if pred >= 1 else 0,
        'fib_target_pct': TIER_TO_PCT.get(int(pred), 0.02),
        'entry': current_price,
        'take_profit': None,
        'stop_loss': None,
    }

    if _is_buy(int(pred), confidence, p_bullish, base_rate_bullish, min_confidence):
        target_pct = TIER_TO_PCT.get(int(pred), 0.03)
        max_sl = getattr(settings, 'MAX_RISK_PER_TRADE', SL_PCT)
        sl_pct = round(min(target_pct, max_sl), 4)
        signal['action'] = 'BUY'
        signal['fib_target_pct'] = target_pct
        signal['tp_pct'] = target_pct
        signal['sl_pct'] = sl_pct
        signal['take_profit'] = round(current_price * (1.0 + target_pct), 4)
        signal['stop_loss'] = round(current_price * (1.0 - sl_pct), 4)
    elif pred == TIER_CHOP and confidence >= min_confidence:
        # Class 0 means "no tier reached" (chop), not a bearish call
        signal['action'] = 'CHOP'
        signal['fib_target_pct'] = None
        signal['order_type'] = 'signal_only'

    return signal