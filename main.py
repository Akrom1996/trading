import os
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
from positions import save_positions, load_positions
from telegram_bot import (
    notify_signal_with_liq, notify_signal, notify_retrain, notify_daily_limit,
    notify_loss_limit, notify_error, notify_start, notify_stop,
    send_message
)

# ── Symbol: set via env var so the SAME image runs any coin ──
# docker run -e SYMBOL=SOL/USDT ...  (see docker-compose.yml)
SYMBOL = os.getenv('SYMBOL', 'ZEC/USDT')

MAX_RISK_PER_TRADE = 0.02
MAX_DAILY_LOSS     = 0.06
MAX_TRADES_PER_DAY = 10
TRAIN_CANDLES      = 2880
LIVE_CANDLES       = 1000
MAX_MODEL_AGE_HRS  = 12
TRAIL_PCT          = 0.01   # trailing-stop distance below peak price


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
    print(f"[{SYMBOL}] Fetching {TRAIN_CANDLES} candles...")
    df = fetch_ohlcv(SYMBOL, '5m', limit=TRAIN_CANDLES)
    df = add_features(df)
    df = add_labels(df)  # training-only: adds forward-looking target/label
    print(f"  Candles loaded : {len(df)}")
    print(f"  From           : {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')}")
    print(f"  To             : {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
    print("Training model...")
    model, scaler, encoder = train_model(df)
    save_model(model, scaler, encoder, symbol=SYMBOL, candles_count=len(df))
    return model, scaler, encoder


def retrain_barrier_and_save():
    df_barrier = fetch_ohlcv(SYMBOL, '5m', limit=TRAIN_CANDLES)
    df_barrier = add_features(df_barrier)
    barrier_model, barrier_scaler, barrier_encoder, label_stats = \
        train_barrier_model(df_barrier, symbol=SYMBOL)
    save_barrier_model(barrier_model, barrier_scaler, barrier_encoder,
                        symbol=SYMBOL, candles_count=len(df_barrier),
                        label_stats=label_stats)

    # Surface class-imbalance directly in Telegram, not just console --
    # this is what silently degraded the bot before (SL_FIRST pinned
    # near 100%) and nobody noticed until price action made it obvious.
    tp_pct = label_stats.get(f"TP_FIRST (+{TP_PCT*100:.1f}%)", {}).get('pct', 0)
    if tp_pct < 5:
        send_message(
            f"⚠️ <b>[{SYMBOL}] Barrier model imbalance warning</b>\n"
            f"TP_FIRST examples are only {tp_pct:.1f}% of training data — "
            f"live probabilities may stay stuck near 0%."
        )
    return barrier_model, barrier_scaler, barrier_encoder


def close_position(pos, current_price, reason, daily_pnl):
    if pos['action'] == 'BUY':
        pnl = ((current_price - pos['entry']) / pos['entry']) * 100
    else:
        pnl = ((pos['entry'] - current_price) / pos['entry']) * 100

    icon = "✅" if pnl >= 0 else "🛑"
    msg = (
        f"{icon} <b>[{SYMBOL}] {pos['action']} Closed — {reason}</b>\n\n"
        f"💰 Entry:  <b>{pos['entry']}</b>\n"
        f"🎯 Close:  <b>{current_price}</b>\n"
        f"📈 PnL:    <b>{pnl:+.2f}%</b>"
    )
    send_message(msg)
    return daily_pnl + pnl


def run_bot(model, scaler, encoder, barrier_model=None, barrier_scaler=None, barrier_encoder=None):
    # ── Restore state from disk if the process was restarted mid-trade ──
    saved_positions, saved_trades, saved_pnl, saved_day = load_positions(SYMBOL)
    now0 = datetime.now()
    if saved_positions or saved_day == now0.day:
        open_positions = saved_positions
        daily_trades   = saved_trades
        daily_pnl      = saved_pnl
        print(f"[{SYMBOL}] Restored {len(open_positions)} open position(s) "
              f"and today's counters from disk")
        if open_positions:
            send_message(f"🔄 <b>[{SYMBOL}] Restarted — restored "
                          f"{len(open_positions)} open position(s) from disk</b>")
    else:
        open_positions = []
        daily_trades   = 0
        daily_pnl      = 0.0

    last_retrain = datetime.now()
    last_day     = datetime.now().day
    limit_notice_sent = False  # only send the "waiting" summary once per limit-reached streak

    while True:
        now = datetime.now()

        # ── Reset daily counters at midnight ─────────────────
        if now.day != last_day:
            daily_trades   = 0
            daily_pnl      = 0.0
            last_day       = now.day
            open_positions = []
            limit_notice_sent = False
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Daily counters reset")
            send_message(f"🔄 <b>[{SYMBOL}] Daily counters reset</b>")
            save_positions(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)

        # ── Retrain every 1 hour ──────────────────────────────
        minutes_since_retrain = (now - last_retrain).seconds / 60
        if minutes_since_retrain >= 60:
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Retraining model...")
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
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Model retrained and saved")
            except Exception as e:
                notify_error(f"[{SYMBOL}] {e}")
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Retrain failed: {e}")

            # ── Retrain barrier (TP magnitude) model too ──────
            try:
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Retraining barrier model...")
                barrier_model, barrier_scaler, barrier_encoder = retrain_barrier_and_save()
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Barrier model retrained and saved")
            except Exception as e:
                notify_error(f"[{SYMBOL}] Barrier retrain failed: {e}")
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Barrier retrain failed: {e}")

        # ── Check open positions: trailing stop + hard SL safety net ──
        try:
            current_price = fetch_current_price(SYMBOL)
            decimals      = get_decimal_places(current_price)

            closed_positions = []
            for pos in open_positions:
                if pos['action'] != 'BUY':
                    continue  # spot/long-only -- shouldn't happen, but be safe

                # Ratchet the trailing stop up as price makes new highs
                if current_price > pos['peak_price']:
                    pos['peak_price'] = current_price
                    candidate_trail = current_price * (1 - TRAIL_PCT)
                    pos['trailing_stop'] = max(pos['trailing_stop'], candidate_trail)

                if current_price <= pos['trailing_stop']:
                    daily_pnl = close_position(pos, current_price, "Trailing Stop Hit", daily_pnl)
                    closed_positions.append(pos)
                elif current_price <= pos['stop_loss']:
                    # Hard safety-net SL -- protects against a fast gap-down
                    # the trailing stop didn't get a chance to ratchet against
                    daily_pnl = close_position(pos, current_price, "Hard SL Hit", daily_pnl)
                    closed_positions.append(pos)

            for pos in closed_positions:
                open_positions.remove(pos)
            if closed_positions:
                save_positions(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)

        except Exception as e:
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Position check error: {e}")

        # ── Skip if limits reached ────────────────────────────
        if daily_trades >= MAX_TRADES_PER_DAY:
            open_count = len(open_positions)
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Max trades reached | "
                  f"Open positions: {open_count} | Price: {current_price}")
            # Send this summary once when the limit is first hit, not every
            # minute -- close_position() already messages Telegram the moment
            # a trailing stop or hard SL actually fires, so a repeated
            # "waiting" ping every 60s was pure noise on top of that.
            if open_count > 0 and not limit_notice_sent:
                pos_summary = "\n".join([
                    f"  🟢 BUY | Entry: {p['entry']} | "
                    f"Peak: {p['peak_price']} | Trail: {p['trailing_stop']:.6g} | "
                    f"Hard SL: {p['stop_loss']}"
                    for p in open_positions
                ])
                send_message(
                    f"⏳ <b>[{SYMBOL}] Daily trade limit reached — "
                    f"{open_count} position(s) still open, will notify on exit</b>\n\n"
                    f"{pos_summary}\n\n"
                    f"💵 Current price: <b>{current_price}</b>\n"
                    f"📊 Daily PnL: <b>{daily_pnl:.2f}%</b>"
                )
                limit_notice_sent = True
            time.sleep(60)
            continue
        else:
            limit_notice_sent = False  # reset once we're back under the limit

        if daily_pnl <= -MAX_DAILY_LOSS:
            notify_loss_limit()
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Daily loss limit hit, stopping for today...")
            time.sleep(3600)
            continue

        # ── Fetch live data and generate signal ───────────────
        try:
            df     = fetch_ohlcv(SYMBOL, '5m', limit=LIVE_CANDLES)
            df     = add_features(df)
            signal = generate_signal(model, scaler, encoder, df)

            coin        = SYMBOL.split('/')[0]
            liq, levels = get_liq_data(coin, current_price)

            # Spot trading: no shorting, so SELL is never a new entry.
            if signal['action'] == 'SELL':
                print(f"  ℹ️  SELL prediction ignored — spot trading is long-only")
                signal['action'] = None

            if signal['action'] and liq:
                if signal['action'] == 'BUY' and liq['bias'] == 'BEARISH':
                    print(f"  ⚠️  BUY blocked — liquidation bias is BEARISH")
                    signal['action'] = None
                if liq['liq_spike']:
                    print(f"  ⚠️  Signal blocked — liquidation spike detected")
                    signal['action'] = None

            if signal['action'] == 'BUY':
                decimals = get_decimal_places(current_price)
                enter    = False

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
                    signal['action'] = None

            if signal['action'] == 'BUY':
                signal['entry'] = current_price
                # Hard SL stays as a safety net; there's no fixed TP anymore
                # -- exits are via trailing stop once price has moved up.
                signal['stop_loss'] = round(current_price * (1 - SL_PCT), decimals)

            liq_info = f"Liq: {liq['bias']} ({liq['liq_ratio']:.0%})" if liq else "Liq: N/A"
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] "
                  f"Signal: {signal['action'] or 'NONE':4} | "
                  f"Confidence: {signal['confidence']:.2f} | "
                  f"Price: {current_price} | "
                  f"{liq_info} | "
                  f"Trades: {daily_trades}/{MAX_TRADES_PER_DAY}")

            if signal['action']:
                level_text = ""
                if levels:
                    level_text = "\n\n🗺 <b>Nearby Liq Levels:</b>\n"
                    for lv in levels[:3]:
                        level_text += (f"  {'⬆️' if lv['direction'] == 'ABOVE' else '⬇️'} "
                                        f"${lv['price']} ({lv['distance']}% away — ${lv['amount']:,.0f})\n")

                notify_signal_with_liq(signal, daily_trades + 1, liq, level_text)
                open_positions.append({
                    'action':        signal['action'],
                    'entry':         signal['entry'],
                    'stop_loss':     signal['stop_loss'],
                    'peak_price':    signal['entry'],
                    'trailing_stop': signal['stop_loss'],
                    'opened_at':     now.strftime('%H:%M'),
                })
                daily_trades += 1
                save_positions(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)

        except Exception as e:
            notify_error(f"[{SYMBOL}] {e}")
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Error: {e}")

        time.sleep(60)


if __name__ == "__main__":
    print("=" * 50)
    print(f"  {SYMBOL} Trading Bot")
    print("=" * 50)

    print("Checking for saved model...")
    model, scaler, encoder, metadata = load_model(symbol=SYMBOL)

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
    barrier_model, barrier_scaler, barrier_encoder, barrier_meta = load_barrier_model(symbol=SYMBOL)
    if barrier_model is None:
        print("  No saved barrier model, training from scratch...")
        barrier_model, barrier_scaler, barrier_encoder = retrain_barrier_and_save()
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