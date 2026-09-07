"""
Triple-barrier probability model -- simplified for spot-only trading.

Since spot trading is long-only (no shorting), there's only one question
that matters: from the current candle, does price reach +TP_PCT before
-SL_PCT within the horizon, or not?

    TP_FIRST  : price hits +TP_PCT before -SL_PCT and before timeout -> BUY
    SL_FIRST  : price hits -SL_PCT before +TP_PCT -> don't enter
    TIMEOUT   : horizon passes with neither touched -> don't enter

CHANGE LOG (Sept 2026):
  - Horizon shortened from 2 days (576 candles) to 12 hours (144 candles).
    A 2-day horizon with a 1% SL / 1.5% TP gave the -1% barrier enormous
    time to get randomly tapped by ordinary chop before a clean +1.5%
    could register -- live logs showed SL_FIRST=100% almost permanently.
  - SL widened relative to TP (was SL=1%/TP=1.5%, now SL=1.5%/TP=1.5%)
    so normal in-trend pullbacks don't auto-disqualify a good setup.
  - Added sample_weight='balanced' so XGBoost can't just default to
    always predicting the majority class when TP_FIRST is rare.
  - Model/label-distribution logging now returns machine-readable data
    so callers (e.g. Telegram notifier) can surface imbalance warnings
    instead of it only being visible in console output.
  - Paths are now symbol-aware so multiple coins running in parallel
    (SOL/ZEC/BTC) don't overwrite each other's saved models.

Uses the same FEATURES your existing model already computes -- no new
indicators needed.
"""

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from trading_model import FEATURES  # reuse your existing feature set

# ── Barrier configuration ──────────────────────────────────
CANDLES_PER_DAY  = 288                  # 24h * 60min / 5min candles
HORIZON_CANDLES  = CANDLES_PER_DAY // 2 # 12-hour forward window (was 2 days)
TP_PCT           = 0.015                # +1.5% target
SL_PCT           = 0.015                # -1.5% stop (was 1% -- too tight)

LABEL_SL_FIRST = 0
LABEL_TP_FIRST = 1
LABEL_TIMEOUT  = 2

LABEL_NAMES = {
    LABEL_SL_FIRST: f"SL_FIRST (-{SL_PCT*100:.1f}%)",
    LABEL_TP_FIRST: f"TP_FIRST (+{TP_PCT*100:.1f}%)",
    LABEL_TIMEOUT:  "TIMEOUT",
}

# Warn if a class falls below this share of training examples -- XGBoost
# can still learn from an imbalanced set, but under ~5% it tends to just
# collapse toward always predicting the majority class.
MIN_HEALTHY_CLASS_SHARE = 0.05


def _model_path(symbol: str) -> str:
    """e.g. 'ZEC/USDT' -> 'barrier_model_ZECUSDT.pkl' -- keeps parallel
    per-coin containers/processes from clobbering each other's models."""
    safe = symbol.replace('/', '').upper()
    return f"barrier_model_{safe}.pkl"


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


def train_barrier_model(df: pd.DataFrame, symbol: str = None):
    """
    df must already have features added (same add_features() pipeline
    you use for the main model) plus 'high', 'low', 'close' columns.

    Returns (model, scaler, encoder, label_stats). label_stats is a dict
    you can log or push to Telegram so imbalance is visible without
    reading console output by hand.
    """
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

    if num_classes < 3:
        missing = set(LABEL_NAMES) - set(encoder.classes_)
        missing_names = [LABEL_NAMES[m] for m in missing]
        print(f"  ⚠️  No examples for: {', '.join(missing_names)} — "
              f"model will always predict 0% for {'these outcomes' if len(missing) > 1 else 'this outcome'}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balanced sample weighting -- without this, XGBoost minimizes
    # log-loss by leaning toward whichever class is most common, which
    # is exactly the "always predicts SL_FIRST" failure mode we saw live.
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_encoded)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
    )
    model.fit(X_scaled, y_encoded, sample_weight=sample_weight)
    return model, scaler, encoder, label_stats


def save_barrier_model(model, scaler, encoder, symbol: str, path: str = None,
                        candles_count=None, label_stats=None):
    path = path or _model_path(symbol)
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


def load_barrier_model(symbol: str, path: str = None):
    path = path or _model_path(symbol)
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
    Spot/long-only decision rule: enter (BUY) only if P(+TP_PCT before
    -SL_PCT) clears the threshold. Returns (should_buy: bool, reason: str).
    """
    p = barrier_probs['p_tp_first']
    if p >= min_profit_prob:
        return True, f"P(+{TP_PCT*100:.1f}% before -{SL_PCT*100:.1f}%)={p:.0%} — entering"
    return False, f"P(+{TP_PCT*100:.1f}% before -{SL_PCT*100:.1f}%)={p:.0%} below {min_profit_prob:.0%} threshold — skipping"