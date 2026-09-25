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
# from positions import save_positions, load_positions
from telegram_bot import (
    notify_signal_with_liq, notify_signal, notify_retrain, notify_daily_limit,
    notify_loss_limit, notify_error, notify_start, notify_stop,
    send_message
)
from db import init_db, save_bot_state, load_bot_state, record_closed_trade

# ── Symbol: set via env var so the SAME image runs any coin ──
# docker run -e SYMBOL=SOL/USDT ...  (see docker-compose.yml)
SYMBOL = os.getenv('SYMBOL', 'ONE/USDT')

MAX_RISK_PER_TRADE = 0.02
MAX_DAILY_LOSS     = 6      # percentage points -- daily_pnl accumulates as e.g. -1.5
                             # for a -1.5% close, not as a 0.015 fraction, so this
                             # must be in the same units or it trips on the first SL
MAX_TRADES_PER_DAY = 10
TRAIN_CANDLES      = 2880
LIVE_CANDLES       = 1000
MAX_MODEL_AGE_HRS  = 12

# ── Pyramiding config (Option A) ──────────────────────────
# Each position uses a fixed TP/SL (from barrier_model's TP_PCT/SL_PCT).
# When price gets within PYRAMID_TRIGGER_PCT of an *unpyramided* open
# position's TP, and the barrier model still confirms bullish, open a
# NEW position at current price with its own fresh TP/SL. Repeat up to
# MAX_PYRAMID_POSITIONS concurrent positions. Each rung is closed
# independently on its own TP or SL.
MAX_PYRAMID_POSITIONS = 6
PYRAMID_TRIGGER_PCT   = 0.002  # trigger when price is within 0.2% of a position's TP


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

    # Surface class-imbalance directly in Telegram, not just console
    fib_tp_pct = sum(stats.get('pct', 0) for name, stats in label_stats.items() if 'TP_FIB' in name)
    if fib_tp_pct < 5:
        send_message(
            f"⚠️ <b>[{SYMBOL}] Barrier model imbalance warning</b>\n"
            f"Profitable Fibonacci examples are only {fib_tp_pct:.1f}% of training data — "
            f"live probabilities may stay low."
        )
    return barrier_model, barrier_scaler, barrier_encoder


def normalize_position(pos: dict, decimals: int) -> dict:
    """
    Backward-compat: older saved positions (from the trailing-stop
    version) have 'peak_price'/'trailing_stop' instead of 'take_profit'.
    Restoring one of those after this rewrite would KeyError the first
    time the pyramid-version code reads pos['take_profit']. Fill in
    whatever's missing so a restart never crashes regardless of which
    version of the bot originally wrote the file.
    """
    if 'take_profit' not in pos:
        pos['take_profit'] = round(pos['entry'] * (1 + TP_PCT), decimals)
    if 'stop_loss' not in pos:
        pos['stop_loss'] = round(pos['entry'] * (1 - SL_PCT), decimals)
    if 'pyramided' not in pos:
        pos['pyramided'] = True  # unknown history -- don't let it spawn a surprise rung
    pos.pop('peak_price', None)
    pos.pop('trailing_stop', None)
    return pos


def close_position(pos, current_price, reason, daily_pnl):
    if pos["action"] == "BUY":
        pnl = ((current_price - pos["entry"]) / pos["entry"]) * 100
    else:
        pnl = ((pos["entry"] - current_price) / pos["entry"]) * 100

    icon = "✅" if pnl >= 0 else "🛑"
    msg = (
        f"{icon} <b>[{SYMBOL}] {pos['action']} Closed — {reason}</b>\n\n"
        f"💰 Entry:  <b>{pos['entry']}</b>\n"
        f"🎯 Close:  <b>{current_price}</b>\n"
        f"📈 PnL:    <b>{pnl:+.2f}%</b>"
    )
    send_message(msg)

    # Record trade into permanent SQLite history
    record_closed_trade(
        symbol=SYMBOL,
        action=pos["action"],
        entry=pos["entry"],
        exit_price=current_price,
        pnl_pct=pnl,
        reason=reason,
    )

    return daily_pnl + pnl


def run_bot(model, scaler, encoder, barrier_model=None, barrier_scaler=None, barrier_encoder=None):
    # ── Restore state from disk if the process was restarted mid-trade ──
    saved_positions, saved_trades, saved_pnl, saved_day = load_bot_state(SYMBOL)
    now0 = datetime.now()
    if saved_positions or saved_day == now0.day:
        open_positions = saved_positions or []
        daily_trades   = saved_trades if saved_day == now0.day else 0
        daily_pnl      = saved_pnl if saved_day == now0.day else 0.0
        # Normalize in case these were saved by an older schema
        if open_positions:
            try:
                _price_for_decimals = fetch_current_price(SYMBOL)
                _decimals = get_decimal_places(_price_for_decimals)
            except Exception:
                _decimals = 4  # reasonable fallback if the API call fails here
            open_positions = [normalize_position(p, _decimals) for p in open_positions]
        print(f"[{SYMBOL}] Restored {len(open_positions)} open position(s) "
              f"and today's counters from SQLite")
        if open_positions:
            send_message(f"🔄 <b>[{SYMBOL}] Restarted — restored "
                          f"{len(open_positions)} open position(s) from SQLite</b>")
            save_bot_state(SYMBOL, open_positions, daily_trades, daily_pnl, now0.day)
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
            limit_notice_sent = False
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Daily counters reset (holding {len(open_positions)} open position(s))")
            holding_msg = f"\nHolding <b>{len(open_positions)}</b> open position(s) into the new day." if open_positions else ""
            send_message(f"🔄 <b>[{SYMBOL}] Daily counters reset</b>{holding_msg}")
            save_bot_state(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)

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
                # probas   = model.predict_proba(X_scaled)
                # avg_conf = probas.max(axis=1).mean()
                # max_conf = probas.max(axis=1).max()
                # notify_retrain(TRAIN_CANDLES, avg_conf, max_conf)
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

        # ── Check open positions: using Candle High/Low to catch wicks ──
        try:
            # Fetch recent 5m candle data to capture the latest High/Low wicks
            df_recent = fetch_ohlcv(SYMBOL, '5m', limit=2)
            latest_candle = df_recent.iloc[-1]
            candle_high   = float(latest_candle['high'])
            candle_low    = float(latest_candle['low'])
            current_price = float(latest_candle['close'])
            decimals      = get_decimal_places(current_price)

            closed_positions = []
            for pos in open_positions:
                if pos['action'] != 'BUY':
                    continue  # spot/long-only -- shouldn't happen, but be safe

                # 1. Check if the candle high wick reached/passed Take Profit
                if candle_high >= pos['take_profit']:
                    # Close at exact TP target price to accurately simulate a limit order fill
                    daily_pnl = close_position(pos, pos['take_profit'], "TP Hit (Candle High)", daily_pnl)
                    closed_positions.append(pos)
                    # A profitable close frees up a trade slot
                    daily_trades = max(0, daily_trades - 1)

                # 2. Check if the candle low wick reached/passed Stop Loss
                elif candle_low <= pos['stop_loss']:
                    # Close at exact SL target price
                    daily_pnl = close_position(pos, pos['stop_loss'], "SL Hit (Candle Low)", daily_pnl)
                    closed_positions.append(pos)
                    # Losses stay counted against the daily limit

            for pos in closed_positions:
                open_positions.remove(pos)
            if closed_positions:
                save_bot_state(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)

            # ── Pyramiding: open a new rung if price is closing in on an
            # existing (unpyramided) position's TP and the barrier model
            # still confirms bullish.
            if (len(open_positions) < MAX_PYRAMID_POSITIONS
                    and daily_trades < MAX_TRADES_PER_DAY
                    and barrier_model is not None):
                for pos in open_positions:
                    if pos['action'] != 'BUY' or pos.get('pyramided'):
                        continue
                    near_tp = current_price >= pos['take_profit'] * (1 - PYRAMID_TRIGGER_PCT)
                    if not near_tp:
                        continue

                    try:
                        df_pyr = fetch_ohlcv(SYMBOL, '5m', limit=LIVE_CANDLES)
                        df_pyr = add_features(df_pyr)
                        barrier_probs = predict_barrier_probabilities(
                            barrier_model, barrier_scaler, barrier_encoder, df_pyr
                        )
                        still_bullish, reason = should_enter(barrier_probs)
                    except Exception as e:
                        print(f"  ⚠️  Pyramid barrier check failed: {e}")
                        still_bullish = False
                        reason = "barrier check failed"

                    pos['pyramided'] = True  # only ever try once per rung, hit or miss

                    if still_bullish:
                        new_entry = current_price
                        rung_tp_pct = barrier_probs.get('recommended_tp', TP_PCT)
                        new_tp    = round(new_entry * (1 + rung_tp_pct), decimals)
                        new_sl    = round(new_entry * (1 - SL_PCT), decimals)
                        open_positions.append({
                            'action':      'BUY',
                            'entry':       new_entry,
                            'take_profit': new_tp,
                            'stop_loss':   new_sl,
                            'pyramided':   False,
                            'opened_at':   now.strftime('%H:%M'),
                        })
                        daily_trades += 1
                        send_message(
                            f"🔼 <b>[{SYMBOL}] Pyramid entry #{len(open_positions)}</b>\n\n"
                            f"💰 Entry: <b>{new_entry}</b>\n"
                            f"🎯 TP:    <b>{new_tp} (+{rung_tp_pct*100:.0f}% Fib)</b>\n"
                            f"🛑 SL:    <b>{new_sl}</b>\n"
                            f"📊 {reason}"
                        )
                        save_bot_state(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)
                    break  # only evaluate one candidate rung per loop tick

        except Exception as e:
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Position check error: {e}")

        # ── Skip if limits reached ────────────────────────────
        if daily_trades >= MAX_TRADES_PER_DAY:
            open_count = len(open_positions)
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Max trades reached | "
                  f"Open positions: {open_count} | Price: {current_price}")
            if open_count > 0 and not limit_notice_sent:
                pos_summary = "\n".join([
                    f"  🟢 BUY | Entry: {p['entry']} | "
                    f"TP: {p['take_profit']} | SL: {p['stop_loss']}"
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

                if len(open_positions) > 0:
                    print(f"  ℹ️  Skipping fresh signal entry — "
                          f"{len(open_positions)} position(s) already open "
                          f"(new entries come from pyramiding only)")
                    signal['action'] = None
                elif barrier_model is not None:
                    try:
                        barrier_probs = predict_barrier_probabilities(
                            barrier_model, barrier_scaler, barrier_encoder, df
                        )
                        enter, reason = should_enter(barrier_probs)
                        rec_tp = barrier_probs.get('recommended_tp', TP_PCT)
                        print(f"  📊 Barrier probs: "
                              f"SL={barrier_probs['p_sl_first']:.0%} "
                              f"FibProfit={barrier_probs['p_tp_first']:.0%} "
                              f"(1%:{barrier_probs.get('p_fib_1', 0):.0%}, "
                              f"2%:{barrier_probs.get('p_fib_2', 0):.0%}, "
                              f"3%:{barrier_probs.get('p_fib_3', 0):.0%}, "
                              f"5%:{barrier_probs.get('p_fib_5', 0):.0%}, "
                              f"8%:{barrier_probs.get('p_fib_8', 0):.0%}, "
                              f"13%:{barrier_probs.get('p_fib_13', 0):.0%}) "
                              f"timeout={barrier_probs['p_timeout']:.0%} — {reason}")
                    except Exception as e:
                        print(f"  ⚠️  Barrier prediction failed: {e}")
                        enter = False
                else:
                    print("  ⚠️  No barrier model loaded — skipping trade")

                if not enter:
                    signal['action'] = None

            if signal['action'] == 'BUY':
                model_tp = signal.get('fib_target_pct', 0.03)
                barrier_tp = barrier_probs.get('recommended_tp', 0.03) if barrier_model is not None else model_tp
                effective_tp = max(model_tp, barrier_tp)

                signal['fib_target_pct'] = effective_tp
                signal['entry']       = current_price
                signal['take_profit'] = round(current_price * (1 + effective_tp), decimals)
                signal['stop_loss']   = round(current_price * (1 - SL_PCT), decimals)

            liq_info = f"Liq: {liq['bias']} ({liq['liq_ratio']:.0%})" if liq else "Liq: N/A"
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] "
                  f"Signal: {signal['action'] or 'NONE':4} | "
                  f"Pred: {signal['pred_label']:+d} | "
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

                notify_signal_with_liq(signal, daily_trades + 1, liq, level_text, symbol=SYMBOL)
                open_positions.append({
                    'action':      signal['action'],
                    'entry':       signal['entry'],
                    'take_profit': signal['take_profit'],
                    'stop_loss':   signal['stop_loss'],
                    'pyramided':   False,
                    'opened_at':   now.strftime('%H:%M'),
                })
                daily_trades += 1
                save_bot_state(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)

        except Exception as e:
            notify_error(f"[{SYMBOL}] {e}")
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Error: {e}")

        time.sleep(60)


if __name__ == "__main__":
    print("=" * 50)
    print(f"  {SYMBOL} Trading Bot")
    print("=" * 50)

    init_db()

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

    # notify_start(SYMBOL)
    print("Bot started. Press Ctrl+C to stop.")
    print("=" * 50)

    try:
        run_bot(model, scaler, encoder, barrier_model, barrier_scaler, barrier_encoder)
    except KeyboardInterrupt:
        notify_stop(SYMBOL)
        print("\nBot stopped by user")