import time
from datetime import datetime
from fetch_data import fetch_ohlcv, fetch_current_price
from feature import add_features, add_labels
from trading_signal import generate_signal
from liquidation import get_liq_data
from trading_model import (
    train_model, save_model, load_model,
    model_is_fresh, FEATURES
)
from barrier_model import (
    train_barrier_model, save_barrier_model, load_barrier_model,
    predict_barrier_probabilities, should_enter, TP_PCT, SL_PCT
)
from telegram_bot import (
    notify_signal_with_liq, notify_signal, notify_retrain, notify_daily_limit,
    notify_loss_limit, notify_error, notify_start, notify_stop,
    send_message
)

MAX_RISK_PER_TRADE = 0.02
MAX_DAILY_LOSS     = 0.06
MAX_TRADES_PER_DAY = 10
TRAIN_CANDLES      = 2880
LIVE_CANDLES       = 1000
MAX_MODEL_AGE_HRS  = 12
SYMBOL             = 'ZEC/USDT'


def get_decimal_places(price: float) -> int:
    if price >= 100:  return 2
    if price >= 1:    return 4
    if price >= 0.01: return 6
    return 8


def calculate_position_size(portfolio_value, entry, stop_loss):
    risk_amount    = portfolio_value * MAX_RISK_PER_TRADE
    price_risk_pct = abs(entry - stop_loss) / entry
    position_size  = risk_amount / price_risk_pct
    return min(position_size, portfolio_value * 0.25)


def retrain_and_save():
    print(f"Fetching {TRAIN_CANDLES} candles...")
    df = fetch_ohlcv(SYMBOL, '5m', limit=TRAIN_CANDLES)
    df = add_features(df)
    df = add_labels(df)  # training-only: adds forward-looking target/label
    print(f"  Candles loaded : {len(df)}")
    print(f"  From           : {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')}")
    print(f"  To             : {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
    print("Training model...")
    model, scaler, encoder = train_model(df)
    save_model(model, scaler, encoder, candles_count=len(df))
    return model, scaler, encoder


def run_bot(model, scaler, encoder, barrier_model=None, barrier_scaler=None, barrier_encoder=None):
    daily_trades   = 0
    daily_pnl      = 0.0
    last_retrain   = datetime.now()
    last_day       = datetime.now().day
    open_positions = []  # list of open trades

    while True:
        now = datetime.now()

        # ── Reset daily counters at midnight ─────────────────
        if now.day != last_day:
            daily_trades   = 0
            daily_pnl      = 0.0
            last_day       = now.day
            open_positions = []
            print(f"[{now.strftime('%H:%M')}] Daily counters reset")
            send_message("🔄 <b>Daily counters reset</b>")

        # ── Retrain every 1 hour ──────────────────────────────
        minutes_since_retrain = (now - last_retrain).seconds / 60
        if minutes_since_retrain >= 60:
            print(f"[{now.strftime('%H:%M')}] Retraining model...")
            try:
                model, scaler, encoder = retrain_and_save()
                last_retrain           = now
                df_test  = fetch_ohlcv(SYMBOL, '5m', limit=200)
                df_test  = add_features(df_test)
                X_scaled = scaler.transform(df_test[FEATURES])
                probas   = model.predict_proba(X_scaled)
                avg_conf = probas.max(axis=1).mean()
                max_conf = probas.max(axis=1).max()
                notify_retrain(TRAIN_CANDLES, avg_conf, max_conf)
                print(f"[{now.strftime('%H:%M')}] Model retrained and saved")
            except Exception as e:
                notify_error(str(e))
                print(f"[{now.strftime('%H:%M')}] Retrain failed: {e}")

            # ── Retrain barrier (TP magnitude) model too ──────
            try:
                print(f"[{now.strftime('%H:%M')}] Retraining barrier model...")
                df_barrier = fetch_ohlcv(SYMBOL, '5m', limit=TRAIN_CANDLES)
                df_barrier = add_features(df_barrier)
                barrier_model, barrier_scaler, barrier_encoder = train_barrier_model(df_barrier)
                save_barrier_model(barrier_model, barrier_scaler, barrier_encoder,
                                    candles_count=len(df_barrier))
                print(f"[{now.strftime('%H:%M')}] Barrier model retrained and saved")
            except Exception as e:
                notify_error(f"Barrier retrain failed: {e}")
                print(f"[{now.strftime('%H:%M')}] Barrier retrain failed: {e}")

        # ── Check open positions against current price ────────
        try:
            current_price = fetch_current_price(SYMBOL)
            decimals      = get_decimal_places(current_price)

            closed_positions = []
            for pos in open_positions:
                if pos['action'] == 'BUY':
                    if current_price >= pos['take_profit']:
                        pnl = ((pos['take_profit'] - pos['entry']) / pos['entry']) * 100
                        msg = (
                            f"✅ <b>BUY Closed — TP Hit</b>\n\n"
                            f"💰 Entry:  <b>{pos['entry']}</b>\n"
                            f"🎯 Close:  <b>{current_price}</b>\n"
                            f"📈 PnL:    <b>+{pnl:.2f}%</b>"
                        )
                        send_message(msg)
                        daily_pnl += pnl
                        closed_positions.append(pos)
                    elif current_price <= pos['stop_loss']:
                        pnl = ((pos['stop_loss'] - pos['entry']) / pos['entry']) * 100
                        msg = (
                            f"🛑 <b>BUY Closed — SL Hit</b>\n\n"
                            f"💰 Entry:  <b>{pos['entry']}</b>\n"
                            f"🎯 Close:  <b>{current_price}</b>\n"
                            f"📉 PnL:    <b>{pnl:.2f}%</b>"
                        )
                        send_message(msg)
                        daily_pnl += pnl
                        closed_positions.append(pos)

                elif pos['action'] == 'SELL':
                    if current_price <= pos['take_profit']:
                        pnl = ((pos['entry'] - pos['take_profit']) / pos['entry']) * 100
                        msg = (
                            f"✅ <b>SELL Closed — TP Hit</b>\n\n"
                            f"💰 Entry:  <b>{pos['entry']}</b>\n"
                            f"🎯 Close:  <b>{current_price}</b>\n"
                            f"📈 PnL:    <b>+{pnl:.2f}%</b>"
                        )
                        send_message(msg)
                        daily_pnl += pnl
                        closed_positions.append(pos)
                    elif current_price >= pos['stop_loss']:
                        pnl = ((pos['entry'] - pos['stop_loss']) / pos['entry']) * 100
                        msg = (
                            f"🛑 <b>SELL Closed — SL Hit</b>\n\n"
                            f"💰 Entry:  <b>{pos['entry']}</b>\n"
                            f"🎯 Close:  <b>{current_price}</b>\n"
                            f"📉 PnL:    <b>{pnl:.2f}%</b>"
                        )
                        send_message(msg)
                        daily_pnl += pnl
                        closed_positions.append(pos)

            # Remove closed positions
            for pos in closed_positions:
                open_positions.remove(pos)

        except Exception as e:
            print(f"[{now.strftime('%H:%M')}] Position check error: {e}")

        # ── Skip if limits reached ────────────────────────────
        if daily_trades >= MAX_TRADES_PER_DAY:
            open_count = len(open_positions)
            print(f"[{now.strftime('%H:%M')}] Max trades reached | "
                  f"Open positions: {open_count} | "
                  f"Price: {current_price}")
            if open_count > 0:
                pos_summary = "\n".join([
                    f"  {'🟢' if p['action'] == 'BUY' else '🔴'} "
                    f"{p['action']} | Entry: {p['entry']} | "
                    f"TP: {p['take_profit']} | SL: {p['stop_loss']}"
                    for p in open_positions
                ])
                send_message(
                    f"⏳ <b>Waiting — {open_count} open position(s)</b>\n\n"
                    f"{pos_summary}\n\n"
                    f"💵 Current price: <b>{current_price}</b>\n"
                    f"📊 Daily PnL: <b>{daily_pnl:.2f}%</b>"
                )
            time.sleep(60)   # check every minute not 1 hour
            continue

        if daily_pnl <= -MAX_DAILY_LOSS:
            notify_loss_limit()
            print(f"[{now.strftime('%H:%M')}] Daily loss limit hit, stopping for today...")
            time.sleep(3600)
            continue

        # ── Fetch live data and generate signal ───────────────
        try:
            df     = fetch_ohlcv(SYMBOL, '5m', limit=LIVE_CANDLES)
            df     = add_features(df)
            signal = generate_signal(model, scaler, encoder, df)

            # ── Liquidation analysis ──────────────────────────
            coin   = SYMBOL.split('/')[0]   # 'ZEC' from 'ZEC/USDT'
            liq, levels = get_liq_data(coin, current_price)

            # ── Spot trading: no shorting, so SELL is never a new entry.
            # A SELL prediction from the direction model just means "no
            # long edge right now" -- it's not an instruction to open
            # a short position. Only BUY signals get evaluated further.
            if signal['action'] == 'SELL':
                print(f"  ℹ️  SELL prediction ignored — spot trading is long-only")
                signal['action'] = None

            # Block signal if liquidation bias conflicts
            if signal['action'] and liq:
                if signal['action'] == 'BUY' and liq['bias'] == 'BEARISH':
                    print(f"  ⚠️  BUY blocked — liquidation bias is BEARISH")
                    signal['action'] = None  # cancel signal

                # Block during liquidation spikes — too dangerous
                if liq['liq_spike']:
                    print(f"  ⚠️  Signal blocked — liquidation spike detected")
                    signal['action'] = None

            if signal['action'] == 'BUY':
                decimals = get_decimal_places(current_price)
                enter    = False

                # ── Barrier model: P(+3% before -3%) over ~2 days.
                # This is the real entry gate now -- the direction
                # model's BUY prediction alone isn't enough to trade on.
                if barrier_model is not None:
                    try:
                        barrier_probs = predict_barrier_probabilities(
                            barrier_model, barrier_scaler, barrier_encoder, df
                        )
                        enter, reason = should_enter(barrier_probs)
                        print(f"  📊 Barrier probs: "
                              f"SL={barrier_probs['p_sl_first']:.0%} "
                              f"TP={barrier_probs['p_tp_first']:.0%} "
                              f"timeout={barrier_probs['p_timeout']:.0%} — {reason}")
                    except Exception as e:
                        print(f"  ⚠️  Barrier prediction failed: {e}")
                        enter = False
                else:
                    print("  ⚠️  No barrier model loaded — skipping trade")

                if not enter:
                    signal['action'] = None  # no barrier edge -> don't buy

            if signal['action'] == 'BUY':
                signal['entry'] = current_price
                signal['take_profit'] = round(current_price * (1 + TP_PCT), decimals)
                signal['stop_loss']   = round(current_price * (1 - SL_PCT), decimals)

            # ── Print with liquidation info ───────────────────
            liq_info = f"Liq: {liq['bias']} ({liq['liq_ratio']:.0%})" if liq else "Liq: N/A"
            print(f"[{now.strftime('%H:%M')}] "
                f"Signal: {signal['action'] or 'NONE':4} | "
                f"Confidence: {signal['confidence']:.2f} | "
                f"Price: {current_price} | "
                f"{liq_info} | "
                f"Trades: {daily_trades}/{MAX_TRADES_PER_DAY}")

            if signal['action']:
                # Add liquidation levels to Telegram notification
                level_text = ""
                if levels:
                    level_text = "\n\n🗺 <b>Nearby Liq Levels:</b>\n"
                    for lv in levels[:3]:
                        level_text += f"  {'⬆️' if lv['direction'] == 'ABOVE' else '⬇️'} ${lv['price']} ({lv['distance']}% away — ${lv['amount']:,.0f})\n"

                notify_signal_with_liq(signal, daily_trades + 1, liq, level_text)
                open_positions.append({
                    'action':      signal['action'],
                    'entry':       signal['entry'],
                    'take_profit': signal['take_profit'],
                    'stop_loss':   signal['stop_loss'],
                    'opened_at':   now.strftime('%H:%M'),
                })
                daily_trades += 1

        except Exception as e:
            notify_error(str(e))
            print(f"[{now.strftime('%H:%M')}] Error: {e}")

        time.sleep(60)


if __name__ == "__main__":
    print("=" * 50)
    print(f"  {SYMBOL} Trading Bot")
    print("=" * 50)

    print("Checking for saved model...")
    model, scaler, encoder, metadata = load_model()

    if model is not None and model_is_fresh(metadata, max_age_hours=MAX_MODEL_AGE_HRS):
        trained_at = metadata['trained_at'].strftime('%Y-%m-%d %H:%M')
        age_hours  = (datetime.now() - metadata['trained_at']).seconds / 3600
        print(f"  Saved model found!")
        print(f"  Trained at : {trained_at}")
        print(f"  Candles    : {metadata['candles_count']}")
        print(f"  Age        : {age_hours:.1f} hours (fresh)")
    else:
        if model is not None:
            print(f"  Model too old, retraining...")
        else:
            print("  No saved model, training from scratch...")
        model, scaler, encoder = retrain_and_save()

    print("Checking model confidence on recent data...")
    df_test  = fetch_ohlcv(SYMBOL, '5m', limit=200)
    df_test  = add_features(df_test)
    X_scaled = scaler.transform(df_test[FEATURES])
    probas   = model.predict_proba(X_scaled)
    print(f"  Avg confidence: {probas.max(axis=1).mean():.2f}")
    print(f"  Max confidence: {probas.max(axis=1).max():.2f}")

    print("Checking for saved barrier model...")
    barrier_model, barrier_scaler, barrier_encoder, barrier_meta = load_barrier_model()
    if barrier_model is None:
        print("  No saved barrier model, training from scratch...")
        df_barrier = fetch_ohlcv(SYMBOL, '5m', limit=TRAIN_CANDLES)
        df_barrier = add_features(df_barrier)
        barrier_model, barrier_scaler, barrier_encoder = train_barrier_model(df_barrier)
        save_barrier_model(barrier_model, barrier_scaler, barrier_encoder, candles_count=len(df_barrier))
    else:
        print(f"  Saved barrier model found (trained {barrier_meta['trained_at']})")

    notify_start()
    print("Bot started. Press Ctrl+C to stop.")
    print("=" * 50)

    try:
        run_bot(model, scaler, encoder, barrier_model, barrier_scaler, barrier_encoder)
    except KeyboardInterrupt:
        notify_stop()
        print("\nBot stopped by user")