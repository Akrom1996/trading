import os
import pickle
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from feature import FIB_TIER_NAMES

FEATURES = [
    'rsi', 'rsi_diff', 'macd', 'stoch',
    'ema_9', 'ema_21', 'ema_50', 'ema_cross',
    'atr', 'atr_ratio', 'bb_high', 'bb_low', 'bb_width',
    'obv', 'vwap', 'vol_ratio', 'vol_ratio_15m',
    'candle_body', 'price_pos', 'volume',
    'return_5m', 'return_15m', 'return_30m', 'return_1h',
    'range_15m',
]


def _paths(symbol: str):
    """Symbol-specific file paths so SOL/ZEC/BTC running in parallel
    (separate containers or one process) don't overwrite each other's
    saved model/scaler/encoder/metadata."""
    safe = symbol.replace('/', '').upper()
    return {
        'model':    f'saved_model_{safe}.pkl',
        'scaler':   f'saved_scaler_{safe}.pkl',
        'encoder':  f'saved_encoder_{safe}.pkl',
        'metadata': f'model_metadata_{safe}.pkl',
    }


def train_model(df):
    print(f"  Label distribution (Fibonacci Tiers):")
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

    scaler   = StandardScaler()
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
        n_jobs=-1
    )
    model.fit(X_scaled, y_encoded, sample_weight=sample_weights)
    return model, scaler, encoder


def save_model(model, scaler, encoder, symbol: str, candles_count=None):
    paths = _paths(symbol)
    with open(paths['model'], 'wb') as f:
        pickle.dump(model, f)
    with open(paths['scaler'], 'wb') as f:
        pickle.dump(scaler, f)
    with open(paths['encoder'], 'wb') as f:
        pickle.dump(encoder, f)
    with open(paths['metadata'], 'wb') as f:
        pickle.dump({
            'trained_at':    datetime.now(),
            'candles_count': candles_count,
            'symbol':        symbol,
        }, f)
    print(f"  Model saved to {paths['model']}")


def load_model(symbol: str):
    paths = _paths(symbol)
    if not os.path.exists(paths['model']) or \
       not os.path.exists(paths['scaler']) or \
       not os.path.exists(paths['metadata']) or \
       not os.path.exists(paths['encoder']):
        return None, None, None, None

    with open(paths['model'], 'rb') as f:
        model = pickle.load(f)
    with open(paths['scaler'], 'rb') as f:
        scaler = pickle.load(f)
    with open(paths['encoder'], 'rb') as f:
        encoder = pickle.load(f)
    with open(paths['metadata'], 'rb') as f:
        metadata = pickle.load(f)

    if getattr(scaler, 'n_features_in_', None) != len(FEATURES):
        print(f"  Feature set changed ({getattr(scaler, 'n_features_in_', '?')} -> {len(FEATURES)}), retraining model...")
        return None, None, None, None

    return model, scaler, encoder, metadata


def model_is_fresh(metadata, max_age_hours=12):
    if metadata is None:
        return False
    age_hours = (datetime.now() - metadata['trained_at']).seconds / 3600
    return age_hours < max_age_hours