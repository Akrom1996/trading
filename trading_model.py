import os
import pickle
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

FEATURES = [
    'rsi', 'rsi_diff', 'macd', 'stoch',
    'ema_9', 'ema_21', 'ema_50', 'ema_cross',
    'atr', 'bb_high', 'bb_low', 'bb_width',
    'obv', 'vwap', 'vol_ratio',
    'candle_body', 'price_pos', 'volume'
]

MODEL_PATH    = 'saved_model.pkl'
SCALER_PATH   = 'saved_scaler.pkl'
METADATA_PATH = 'model_metadata.pkl'
ENCODER_PATH  = 'saved_encoder.pkl'


def train_model(df):
    print(f"  Label distribution:\n{df['label'].value_counts()}")

    X = df[FEATURES]
    y = df['label']

    # XGBoost needs labels 0,1,2 not -1,0,1
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y_encoded, sample_weight=sample_weights)
    return model, scaler, encoder


def save_model(model, scaler, encoder, candles_count):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump({
            'trained_at':    datetime.now(),
            'candles_count': candles_count,
        }, f)
    print(f"  Model saved to {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH) or \
       not os.path.exists(SCALER_PATH) or \
       not os.path.exists(METADATA_PATH) or \
       not os.path.exists(ENCODER_PATH):
        return None, None, None, None

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)

    return model, scaler, encoder, metadata


def model_is_fresh(metadata, max_age_hours=12):
    if metadata is None:
        return False
    age_hours = (datetime.now() - metadata['trained_at']).seconds / 3600
    return age_hours < max_age_hours