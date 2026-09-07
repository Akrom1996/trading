from trading_model import FEATURES

def generate_signal(model, scaler, encoder, df, min_confidence=0.50):
    latest     = df[FEATURES].iloc[-1:].copy()
    scaled     = scaler.transform(latest)

    proba      = model.predict_proba(scaled)[0]
    pred_enc   = model.predict(scaled)[0]
    confidence = max(proba)

    # Decode back to -1, 0, 1
    pred = encoder.inverse_transform([pred_enc])[0]

    current_price = df['close'].iloc[-1]

    signal = {
        'action':      None,
        'confidence':  confidence,
        'pred_label':  int(pred),
        'entry':       current_price,
        'take_profit': None,
        'stop_loss':   None,
    }

    if confidence >= min_confidence:
        if pred == 1:
            signal['action']      = 'BUY'
            signal['take_profit'] = round(current_price * 1.016, 2)
            signal['stop_loss']   = round(current_price * 0.988, 2)
        elif pred == -1:
            signal['action']      = 'SELL'
            signal['take_profit'] = round(current_price * 0.984, 2)
            signal['stop_loss']   = round(current_price * 1.012, 2)

    return signal