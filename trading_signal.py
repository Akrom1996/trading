from trading_model import FEATURES
from feature import FIBONACCI_LEVELS, FIB_TIER_NAMES

TIER_TO_PCT = {
    0: 0.0,
    1: 0.01,
    2: 0.02,
    3: 0.03,
    4: 0.05,
    5: 0.08,
    6: 0.13,
}

def generate_signal(model, scaler, encoder, df, min_confidence=0.40):
    latest     = df[FEATURES].iloc[-1:].copy()
    scaled     = scaler.transform(latest)

    proba      = model.predict_proba(scaled)[0]
    pred_enc   = model.predict(scaled)[0]
    confidence = float(max(proba))

    # Decode predicted label
    pred = encoder.inverse_transform([pred_enc])[0]

    # Calculate cumulative bullish probability across all positive Fibonacci tiers (>=1)
    class_probs = {cls: prob for cls, prob in zip(encoder.classes_, proba)}
    p_bullish = sum(prob for cls, prob in class_probs.items() if cls >= 1)

    current_price = df['close'].iloc[-1]

    signal = {
        'action':         None,
        'confidence':     confidence,
        'p_bullish':      round(p_bullish, 4),
        'pred_label':     int(pred),
        'fib_tier':       int(pred) if pred >= 1 else 0,
        'fib_target_pct': TIER_TO_PCT.get(int(pred), 0.02),
        'entry':          current_price,
        'take_profit':    None,
        'stop_loss':      None,
    }

    # Signal BUY if predicted class is a Fibonacci tier (>=1) and either class confidence or total bullish probability clears threshold
    if pred >= 1 and (confidence >= min_confidence or p_bullish >= 0.45):
        target_pct = TIER_TO_PCT.get(int(pred), 0.03)
        sl_pct     = 0.02
        signal['action']      = 'BUY'
        signal['take_profit'] = round(current_price * (1.0 + target_pct), 4)
        signal['stop_loss']   = round(current_price * (1.0 - sl_pct), 4)
    elif pred == -1 and confidence >= min_confidence:
        # Legacy support if an older 3-class model is loaded
        signal['action']      = 'SELL'
        signal['take_profit'] = round(current_price * 0.98, 4)
        signal['stop_loss']   = round(current_price * 1.02, 4)

    return signal