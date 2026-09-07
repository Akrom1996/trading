"""
Triple-barrier probability model -- simplified for spot-only trading.

Since spot trading is long-only (no shorting), there's only one question
that matters: from the current candle, does price reach +1.5% before -1%
within the horizon, or not?

    TP_FIRST  : price hits +1.5% before -1% and before timeout -> BUY
    SL_FIRST  : price hits -1% before +1.5% -> don't enter
    TIMEOUT   : horizon passes with neither touched -> don't enter

Uses the same FEATURES your existing model already computes -- no new
indicators needed.
"""

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

from trading_model import FEATURES  # reuse your existing feature set

# ── Barrier configuration ──────────────────────────────────
CANDLES_PER_DAY  = 288                 # 24h * 60min / 5min candles
HORIZON_CANDLES  = CANDLES_PER_DAY * 2 # 2-day forward window
TP_PCT           = 0.015                # +1.5% target
SL_PCT           = 0.01                 # -1% stop

LABEL_SL_FIRST = 0
LABEL_TP_FIRST = 1
LABEL_TIMEOUT  = 2

LABEL_NAMES = {
    LABEL_SL_FIRST: "SL_FIRST (-1%)",
    LABEL_TP_FIRST: "TP_FIRST (+1.5%)",
    LABEL_TIMEOUT:  "TIMEOUT",
}

MODEL_PATH = "barrier_model.pkl"


def label_triple_barrier(df: pd.DataFrame, horizon=HORIZON_CANDLES,
                          sl_pct=SL_PCT, tp_pct=TP_PCT):
    """
    Adds a 'barrier_label' column to df. Requires 'high', 'low', 'close'
    columns. Drops the trailing `horizon` rows since they don't have a
    full forward window to check yet -- return value is shorter than df.
    """
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    n      = len(df)

    labels = np.full(n, -1, dtype=int)

    for i in range(n - horizon):
        entry    = closes[i]
        sl_price = entry * (1 - sl_pct)
        tp_price = entry * (1 + tp_pct)

        label = LABEL_TIMEOUT
        for j in range(i + 1, i + 1 + horizon):
            if lows[j] <= sl_price:
                label = LABEL_SL_FIRST
                break
            if highs[j] >= tp_price:
                label = LABEL_TP_FIRST
                break
        labels[i] = label

    out = df.iloc[:n - horizon].copy()
    out['barrier_label'] = labels[:n - horizon]
    return out


def train_barrier_model(df: pd.DataFrame):
    """
    df must already have features added (same add_features() pipeline
    you use for the main model) plus 'high', 'low', 'close' columns.

    Returns (model, scaler, encoder). LabelEncoder compresses whatever
    labels ARE present into a contiguous range so XGBoost doesn't choke
    if one outcome never occurred in the training window.
    """
    labeled = label_triple_barrier(df)

    print(f"  Label distribution:")
    for lbl, name in LABEL_NAMES.items():
        count = (labeled['barrier_label'] == lbl).sum()
        pct = count / len(labeled) * 100
        print(f"    {name:20s}: {count:5d} ({pct:.1f}%)")

    X = labeled[FEATURES]
    y = labeled['barrier_label']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    if num_classes < 3:
        missing = set(LABEL_NAMES) - set(encoder.classes_)
        missing_names = [LABEL_NAMES[m] for m in missing]
        print(f"  ⚠️  No examples for: {', '.join(missing_names)} — "
              f"model will always predict 0% for {'these outcomes' if len(missing) > 1 else 'this outcome'}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
    )
    model.fit(X_scaled, y_encoded)
    return model, scaler, encoder


def save_barrier_model(model, scaler, encoder, path=MODEL_PATH, candles_count=None):
    with open(path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'encoder': encoder,
            'trained_at': datetime.now(),
            'candles_count': candles_count,
        }, f)


def load_barrier_model(path=MODEL_PATH):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler'], data['encoder'], data
    except FileNotFoundError:
        return None, None, None, None


def predict_barrier_probabilities(model, scaler, encoder, df_latest: pd.DataFrame):
    """
    df_latest: dataframe with features already added, at least 1 row.
    Uses the LAST row (most recent candle) for prediction.

    Maps predict_proba's output back onto the full 3-outcome space so
    a missing training class (e.g. no TP_FIRST examples) reports a
    clean 0% instead of crashing or being silently dropped.
    """
    X = df_latest[FEATURES].iloc[[-1]]
    X_scaled = scaler.transform(X)
    raw_probs = model.predict_proba(X_scaled)[0]

    full_probs = {lbl: 0.0 for lbl in LABEL_NAMES}
    for col_idx, original_label in enumerate(encoder.classes_):
        full_probs[original_label] = float(raw_probs[col_idx])

    return {
        'p_sl_first': round(full_probs[LABEL_SL_FIRST], 4),
        'p_tp_first': round(full_probs[LABEL_TP_FIRST], 4),
        'p_timeout':  round(full_probs[LABEL_TIMEOUT], 4),
    }


def should_enter(barrier_probs: dict, min_profit_prob=0.40):
    """
    Spot/long-only decision rule: enter (BUY) only if P(+1.5% before -1%)
    clears the threshold. Returns (should_buy: bool, reason: str).
    """
    p = barrier_probs['p_tp_first']
    if p >= min_profit_prob:
        return True, f"P(+1.5% before -1%)={p:.0%} — entering"
    return False, f"P(+1.5% before -1%)={p:.0%} below {min_profit_prob:.0%} threshold — skipping"