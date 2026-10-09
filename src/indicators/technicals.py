"""
Technical indicators, ML feature engineering, Fibonacci barriers, model training, and signal generation.
Purely consolidated in src/indicators/technicals.py.
"""

from __future__ import annotations
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    import numpy as np
    import pandas as pd
    import ta
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    HAS_ML_DEPS = True
except ImportError:
    np = None
    pd = None
    ta = None
    HAS_ML_DEPS = False

from config.settings import settings

# ── Fibonacci Tiers & Horizons ─────────────────────────────────
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

TIER_TO_PCT = {
    0: 0.0,
    1: 0.01,
    2: 0.02,
    3: 0.03,
    4: 0.05,
    5: 0.08,
    6: 0.13,
}

LABEL_HORIZON_CANDLES = 144
DEFAULT_SL_PCT = 0.02
DEFAULT_TP_PCT = 0.03
TP_PCT = DEFAULT_TP_PCT
SL_PCT = DEFAULT_SL_PCT

LABEL_SL_FIRST = 0
LABEL_FIB_1 = 1
LABEL_FIB_2 = 2
LABEL_FIB_3 = 3
LABEL_FIB_5 = 4
LABEL_FIB_8 = 5
LABEL_FIB_13 = 6
LABEL_TIMEOUT = 7
LABEL_TP_FIRST = LABEL_FIB_2

LABEL_NAMES = {
    LABEL_SL_FIRST: f"SL_FIRST (-{SL_PCT*100:.1f}%)",
    LABEL_FIB_1: "TP_FIB_1 (+1%)",
    LABEL_FIB_2: "TP_FIB_2 (+2%)",
    LABEL_FIB_3: "TP_FIB_3 (+3%)",
    LABEL_FIB_5: "TP_FIB_5 (+5%)",
    LABEL_FIB_8: "TP_FIB_8 (+8%)",
    LABEL_FIB_13: "TP_FIB_13 (+13%)",
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
    df['vwap'] = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
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


def add_labels(df: pd.DataFrame, horizon: int = LABEL_HORIZON_CANDLES, sl_pct: float = DEFAULT_SL_PCT) -> pd.DataFrame:
    """Labels forward Fibonacci expansion tiers reached before SL."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    labels = np.zeros(n, dtype=int)
    max_gains = np.zeros(n, dtype=float)

    for i in range(n - horizon):
        entry = closes[i]
        sl_price = entry * (1.0 - sl_pct)

        peak_gain = 0.0
        for j in range(i + 1, i + 1 + horizon):
            if lows[j] <= sl_price:
                break
            gain = (highs[j] - entry) / entry
            if gain > peak_gain:
                peak_gain = gain

        max_gains[i] = peak_gain

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
    out['label'] = labels[:n - horizon]
    return out.reset_index(drop=True)


def label_triple_barrier(df: pd.DataFrame, horizon: int = LABEL_HORIZON_CANDLES, sl_pct: float = SL_PCT, tp_pct: float = DEFAULT_TP_PCT) -> pd.DataFrame:
    """Triple-barrier labeling for spot trades."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    labels = np.full(n, -1, dtype=int)

    for i in range(n - horizon):
        entry = closes[i]
        sl_price = entry * (1.0 - sl_pct)
        stopped_out = False
        peak_gain = 0.0

        for j in range(i + 1, i + 1 + horizon):
            if lows[j] <= sl_price:
                stopped_out = True
                break
            gain = (highs[j] - entry) / entry
            if gain > peak_gain:
                peak_gain = gain

        if peak_gain >= 0.13:
            label = LABEL_FIB_13
        elif peak_gain >= 0.08:
            label = LABEL_FIB_8
        elif peak_gain >= 0.05:
            label = LABEL_FIB_5
        elif peak_gain >= 0.03:
            label = LABEL_FIB_3
        elif peak_gain >= 0.02:
            label = LABEL_FIB_2
        elif peak_gain >= 0.01:
            label = LABEL_FIB_1
        elif stopped_out:
            label = LABEL_SL_FIRST
        else:
            label = LABEL_TIMEOUT

        labels[i] = label

    out = df.iloc[:n - horizon].copy()
    out['barrier_label'] = labels[:n - horizon]
    return out


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


def train_model(df: pd.DataFrame) -> Tuple[Any, Any, Any]:
    print("  Label distribution (Fibonacci Tiers):")
    total = len(df)
    for tier, name in FIB_TIER_NAMES.items():
        count = int((df['label'] == tier).sum())
        pct = (count / total * 100) if total else 0.0
        if count > 0:
            print(f"    Tier {tier} ({name:22s}): {count:5d} ({pct:.1f}%)")

    X = df[FEATURES]
    y = df['label']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_encoded)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    is_multiclass = num_classes > 2
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob' if is_multiclass else 'binary:logistic',
        num_class=num_classes if is_multiclass else None,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_encoded, sample_weight=sample_weights)
    return model, scaler, encoder


def save_model(model: Any, scaler: Any, encoder: Any, symbol: str, candles_count: Optional[int] = None):
    paths = _get_model_paths(symbol)
    with open(paths['model'], 'wb') as f:
        pickle.dump(model, f)
    with open(paths['scaler'], 'wb') as f:
        pickle.dump(scaler, f)
    with open(paths['encoder'], 'wb') as f:
        pickle.dump(encoder, f)
    with open(paths['metadata'], 'wb') as f:
        pickle.dump({
            'trained_at': datetime.now(),
            'candles_count': candles_count,
            'symbol': symbol,
        }, f)
    print(f"  Model saved to {paths['model']}")


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


def train_barrier_model(df: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    labeled = label_triple_barrier(df)
    total = len(labeled)
    label_stats = {}
    print(f"  Label distribution{f' ({symbol})' if symbol else ''}:")
    for lbl, name in LABEL_NAMES.items():
        count = int((labeled['barrier_label'] == lbl).sum())
        pct = count / total * 100 if total else 0.0
        label_stats[name] = {'count': count, 'pct': round(pct, 1)}
        flag = " ⚠️ low" if pct / 100 < MIN_HEALTHY_CLASS_SHARE else ""
        print(f"    {name:20s}: {count:5d} ({pct:.1f}%){flag}")

    X = labeled[FEATURES]
    y = labeled['barrier_label']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_encoded)

    is_multiclass = num_classes > 2
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective='multi:softprob' if is_multiclass else 'binary:logistic',
        num_class=num_classes if is_multiclass else None,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_encoded, sample_weight=sample_weight)
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
    p_fib_13 = full_probs[LABEL_FIB_13]

    p_profitable = p_fib_1 + p_fib_2 + p_fib_3 + p_fib_5 + p_fib_8 + p_fib_13
    p_extended = p_fib_3 + p_fib_5 + p_fib_8 + p_fib_13

    recommended_tp = 0.02
    if p_fib_13 >= 0.12:
        recommended_tp = 0.13
    elif (p_fib_8 + p_fib_13) >= 0.15:
        recommended_tp = 0.08
    elif (p_fib_5 + p_fib_8 + p_fib_13) >= 0.20:
        recommended_tp = 0.05
    elif p_extended >= 0.25:
        recommended_tp = 0.03
    elif p_profitable >= 0.35:
        recommended_tp = 0.02

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
        'p_fib_13': round(p_fib_13, 4),
        'recommended_tp': recommended_tp,
    }


def should_enter(barrier_probs: Dict[str, Any], min_profit_prob: float = 0.35) -> Tuple[bool, str]:
    p = barrier_probs['p_tp_first']
    rec_tp = barrier_probs.get('recommended_tp', DEFAULT_TP_PCT)
    if p >= min_profit_prob:
        return True, f"P(Fib profit before -{SL_PCT*100:.1f}%)={p:.0%} (Target: +{rec_tp*100:.0f}%) — entering"
    return False, f"P(Fib profit before -{SL_PCT*100:.1f}%)={p:.0%} below {min_profit_prob:.0%} threshold — skipping"


# ── Signal Generation ───────────────────────────────────────────

def generate_signal(model: Any, scaler: Any, encoder: Any, df: pd.DataFrame, min_confidence: float = 0.40) -> Dict[str, Any]:
    latest = df[FEATURES].iloc[-1:].copy()
    scaled = scaler.transform(latest)

    proba = model.predict_proba(scaled)[0]
    pred_enc = model.predict(scaled)[0]
    confidence = float(max(proba))

    pred = encoder.inverse_transform([pred_enc])[0]
    class_probs = {cls: prob for cls, prob in zip(encoder.classes_, proba)}
    p_bullish = sum(prob for cls, prob in class_probs.items() if cls >= 1)

    current_price = float(df['close'].iloc[-1])

    signal = {
        'action': None,
        'confidence': confidence,
        'p_bullish': round(p_bullish, 4),
        'pred_label': int(pred),
        'fib_tier': int(pred) if pred >= 1 else 0,
        'fib_target_pct': TIER_TO_PCT.get(int(pred), 0.02),
        'entry': current_price,
        'take_profit': None,
        'stop_loss': None,
    }

    if pred >= 1 and (confidence >= min_confidence or p_bullish >= 0.45):
        target_pct = TIER_TO_PCT.get(int(pred), 0.03)
        sl_pct = 0.02
        signal['action'] = 'BUY'
        signal['take_profit'] = round(current_price * (1.0 + target_pct), 4)
        signal['stop_loss'] = round(current_price * (1.0 - sl_pct), 4)
    elif pred == -1 and confidence >= min_confidence:
        signal['action'] = 'SELL'
        signal['take_profit'] = round(current_price * 0.98, 4)
        signal['stop_loss'] = round(current_price * 1.02, 4)

    return signal
