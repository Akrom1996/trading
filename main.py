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
from telegram_bot import (
    notify_signal_with_liq, notify_signal, notify_retrain, notify_daily_limit,
    notify_loss_limit, notify_error, notify_start, notify_stop,
    send_message
)
from db import get_last_tp_time, init_db, save_bot_state, load_bot_state, record_closed_trade, get_last_sl_time
from orders import (
    place_limit_buy, place_market_sell, place_oco_sell,
    check_order_status, cancel_order,
)

# ── Symbol: set via env var so the SAME image runs any coin ──
SYMBOL = os.getenv('SYMBOL', 'ZEC/USDT')

MAX_RISK_PER_TRADE = 0.02
MAX_DAILY_LOSS     = 6      # percentage points -- daily_pnl accumulates as e.g. -1.5
TRAIN_CANDLES      = 2880
LIVE_CANDLES       = 1000
MAX_MODEL_AGE_HRS  = 12

MAX_PYRAMID_POSITIONS = 6
PYRAMID_TRIGGER_PCT   = 0.002

SL_COOLDOWN_MINUTES = float(os.getenv('SL_COOLDOWN_MINUTES', '30'))
TP_COOLDOWN_MINUTES = float(os.getenv('TP_COOLDOWN_MINUTES', '10'))
MAX_TRADES_PER_DAY  = int(os.getenv('MAX_TRADES_PER_DAY', '10'))
QUOTE_AMOUNT_PER_TRADE = float(os.getenv('QUOTE_AMOUNT_PER_TRADE', '20'))
ENTRY_OFFSET_PCT    = float(os.getenv('ENTRY_OFFSET_PCT', '0.003'))

# A limit buy placed below market may never fill. Give up and cancel
# after this long, rather than leaving a pending position forever.
PENDING_FILL_TIMEOUT_MINUTES = 15


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
    df = add_labels(df)
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
    Backward-compat for positions saved by older code paths. Fills in
    whatever's missing so a restart never crashes regardless of which
    version originally wrote the record.
    """
    if 'take_profit' not in pos:
        pos['take_profit'] = round(pos['entry'] * (1 + TP_PCT), decimals)
    if 'stop_loss' not in pos:
        pos['stop_loss'] = round(pos['entry'] * (1 - SL_PCT), decimals)
    if 'pyramided' not in pos:
        pos['pyramided'] = True
    if 'status' not in pos:
        # Older records predate pending-fill tracking -- treat as already
        # open (best-effort; can't retroactively know if it truly filled).
        pos['status'] = 'open'
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

    # If an OCO sell is live on the exchange for this position, the
    # exchange has ALREADY executed the matching leg (or will, as soon
    # as price touches it) -- placing our own market sell on top would
    # double-sell. Only place a manual sell here as a fallback for
    # positions that never got OCO protection (e.g. it failed to place).
    if not pos.get('oco_order_id') and pos.get('base_amount'):
        try:
            place_market_sell(SYMBOL, pos['base_amount'])
        except Exception as e:
            notify_error(f"[{SYMBOL}] SELL order failed on close: {e}")
            print(f"[{SYMBOL}] Order error on close: {e}")

    record_closed_trade(
        symbol=SYMBOL, action=pos["action"], entry=pos["entry"],
        exit_price=current_price, pnl_pct=pnl, reason=reason,
    )
    return daily_pnl + pnl


def process_pending_fill(pos, now):
    """
    ONE non-blocking check of a pending limit-buy's fill status. Returns
    'still_pending', 'filled', or 'gave_up' -- caller decides what to do
    next. Never sleeps or loops -- called once per main loop tick.
    """
    status_info = check_order_status(SYMBOL, pos['buy_order_id'])
    status = status_info['status']

    if status == 'closed':
        pos['status']      = 'open'
        pos['base_amount'] = status_info['filled_amount'] or pos.get('base_amount')
        if status_info['avg_price']:
            pos['entry'] = status_info['avg_price']  # real fill price, not the limit we asked for
        return 'filled'

    if status == 'canceled':
        return 'gave_up'

    # still 'open'/'unknown' -- check timeout
    placed_at = pos.get('order_placed_at')
    if placed_at and (now - placed_at).total_seconds() > PENDING_FILL_TIMEOUT_MINUTES * 60:
        cancel_order(SYMBOL, pos['buy_order_id'])
        return 'gave_up'

    return 'still_pending'


def open_position_with_limit_buy(entry_price, take_profit, stop_loss, now, label):
    """
    Places the limit buy and returns a new position dict.
    If the order fails (e.g., Insufficient Funds), returns a position with status 'failed'
    so state/DB can be recorded, preventing constantTelegram spam every minute.
    """
    now_iso = now.isoformat() if isinstance(now, datetime) else now
    
    pos = {
        'action':          'BUY',
        'entry':           entry_price,
        'take_profit':     take_profit,
        'stop_loss':       stop_loss,
        'pyramided':       False,
        'status':          'pending_fill',
        'buy_order_id':    None,
        'order_placed_at': now_iso,
        'base_amount':     0.0,
        'oco_order_id':    None,
        'opened_at':       now.strftime('%H:%M') if isinstance(now, datetime) else now,
    }

    try:
        buy_result = place_limit_buy(SYMBOL, entry_price, QUOTE_AMOUNT_PER_TRADE)
        pos['buy_order_id']  = buy_result.get('id')
        pos['base_amount']   = buy_result.get('filled_amount')

        # if buy_result.get('status') == 'closed':
        #     pos['status'] = 'open'

        return pos

    except Exception as e:
        # Mark as failed so it gets recorded in DB and stops retrying
        pos['status'] = 'failed'
        pos['error']  = str(e)
        
        notify_error(f"❌ [{SYMBOL}] {label} BUY failed (Insufficient funds or exchange error): {e}")
        print(f"[{SYMBOL}] Order placement failed: {e}")
        
        return pos
    

def run_bot(model, scaler, encoder, barrier_model=None, barrier_scaler=None, barrier_encoder=None):
    saved_positions, saved_trades, saved_pnl, saved_day = load_bot_state(SYMBOL)
    now0 = datetime.now()
    if saved_positions or saved_day == now0.day:
        open_positions = saved_positions or []
        daily_trades   = saved_trades if saved_day == now0.day else 0
        daily_pnl      = saved_pnl if saved_day == now0.day else 0.0
        if open_positions:
            try:
                _price_for_decimals = fetch_current_price(SYMBOL)
                _decimals = get_decimal_places(_price_for_decimals)
            except Exception:
                _decimals = 4
            open_positions = [normalize_position(p, _decimals) for p in open_positions]
        print(f"[{SYMBOL}] Restored {len(open_positions)} open position(s) and today's counters from SQLite")
        if open_positions:
            send_message(f"🔄 <b>[{SYMBOL}] Restarted — restored {len(open_positions)} open position(s) from DB</b>")
            save_bot_state(SYMBOL, open_positions, daily_trades, daily_pnl, now0.day)
    else:
        open_positions = []
        daily_trades   = 0
        daily_pnl      = 0.0

    last_retrain = datetime.now()
    last_day     = datetime.now().day
    limit_notice_sent = False

    last_sl_time = get_last_sl_time(SYMBOL)
    if last_sl_time is not None:
        secs_since_sl = (now0 - last_sl_time).total_seconds()
        if secs_since_sl < SL_COOLDOWN_MINUTES * 60:
            mins_left = int((SL_COOLDOWN_MINUTES * 60 - secs_since_sl) // 60) + 1
            print(f"[{SYMBOL}] Active Stop Loss cooldown restored: {mins_left}m remaining")
        else:
            last_sl_time = None

    last_tp_time = get_last_tp_time(SYMBOL)
    if last_tp_time is not None:
        secs_since_tp = (now0 - last_tp_time).total_seconds()
        if secs_since_tp < TP_COOLDOWN_MINUTES * 60:
            mins_left = int((TP_COOLDOWN_MINUTES * 60 - secs_since_tp) // 60) + 1
            print(f"[{SYMBOL}] Active TP cooldown restored: {mins_left}m remaining")
        else:
            last_tp_time = None

    while True:
        now = datetime.now()

        # Reset daily counters at midnight
        if now.day != last_day:
            daily_trades   = 0
            daily_pnl      = 0.0
            last_day       = now.day
            limit_notice_sent = False
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Daily counters reset (holding {len(open_positions)} open position(s))")
            holding_msg = f"\nHolding <b>{len(open_positions)}</b> open position(s) into the new day." if open_positions else ""
            send_message(f"🔄 <b>[{SYMBOL}] Daily counters reset</b>{holding_msg}")
            save_bot_state(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)

        # Retrain every 1 hour
        minutes_since_retrain = (now - last_retrain).total_seconds() / 60
        if minutes_since_retrain >= 60:
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Retraining model...")
            try:
                model, scaler, encoder = retrain_and_save()
                last_retrain           = now
                df_test  = fetch_ohlcv(SYMBOL, '5m', limit=200)
                df_test  = add_features(df_test)
                X_scaled = scaler.transform(df_test[FEATURES])
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Model retrained and saved")
            except Exception as e:
                notify_error(f"[{SYMBOL}] {e}")
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Retrain failed: {e}")

            try:
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Retraining barrier model...")
                barrier_model, barrier_scaler, barrier_encoder = retrain_barrier_and_save()
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Barrier model retrained and saved")
            except Exception as e:
                notify_error(f"[{SYMBOL}] Barrier retrain failed: {e}")
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Barrier retrain failed: {e}")

        try:
            df_recent = fetch_ohlcv(SYMBOL, '5m', limit=2)
            latest_candle = df_recent.iloc[-1]
            candle_high   = float(latest_candle['high'])
            candle_low    = float(latest_candle['low'])
            current_price = float(latest_candle['close'])
            decimals      = get_decimal_places(current_price)

            # ── Step 1: advance any pending limit-buy orders ──────
            # ONE non-blocking check per position per tick -- no waiting.
            still_pending_positions = []
            for pos in open_positions:
                if pos.get('status') != 'pending_fill':
                    still_pending_positions.append(pos)
                    continue

                outcome = process_pending_fill(pos, now)
                if outcome == 'still_pending':
                    still_pending_positions.append(pos)
                elif outcome == 'gave_up':
                    print(f"[{SYMBOL}] Limit buy for position opened at "
                          f"{pos.get('opened_at')} never filled — giving up")
                    send_message(f"⏱️ <b>[{SYMBOL}] Limit buy never filled, canceled</b>\n"
                                 f"Target entry: {pos['entry']}")
                    daily_trades = max(0, daily_trades - 1)  # give the slot back
                    # not added to still_pending_positions -> effectively removed
                elif outcome == 'filled':
                    print(f"[{SYMBOL}] ✅ Limit buy filled @ {pos['entry']} — placing OCO")
                    try:
                        oco_result = place_oco_sell(
                            SYMBOL, pos['base_amount'], pos['take_profit'], pos['stop_loss']
                        )
                        pos['oco_order_id'] = oco_result.get('id') or oco_result.get('orderListId')
                    except Exception as e:
                        # Buy filled but OCO failed -- position is REAL and
                        # UNPROTECTED on the exchange. This is the single
                        # worst state to be in silently, so alert loudly.
                        # The bot's own candle-based TP/SL check below still
                        # covers this position as a fallback.
                        notify_error(f"🚨 [{SYMBOL}] UNPROTECTED POSITION — buy filled but "
                                     f"OCO failed: {e}. Bot's own TP/SL check is the only "
                                     f"protection until this is placed manually or retried.")
                        print(f"[{SYMBOL}] OCO placement failed: {e}")
                    still_pending_positions.append(pos)

            open_positions = still_pending_positions

            # ── Step 2: check OPEN positions' TP/SL via candle high/low ──
            # This remains active even for OCO-protected positions -- it's
            # what updates our own records/cooldowns/daily_pnl. The OCO on
            # the exchange is the actual execution guarantee; this is our
            # bookkeeping of what already happened (or a fallback sell if
            # OCO placement failed above).
            closed_positions = []
            for pos in open_positions:
                if pos.get('status') != 'open' or pos['action'] != 'BUY':
                    continue

                if candle_high >= pos['take_profit']:
                    daily_pnl = close_position(pos, pos['take_profit'], "TP Hit (Candle High)", daily_pnl)
                    closed_positions.append(pos)
                    last_tp_time = now
                    print(f"⏸️ [{SYMBOL}] TP Cooldown Initiated ({TP_COOLDOWN_MINUTES} minutes)")

                elif candle_low <= pos['stop_loss']:
                    daily_pnl = close_position(pos, pos['stop_loss'], "SL Hit (Candle Low)", daily_pnl)
                    closed_positions.append(pos)
                    last_sl_time = now
                    print(f"⏸️ [{SYMBOL}] Stop Loss Cooldown Initiated ({SL_COOLDOWN_MINUTES} minutes)")

            for pos in closed_positions:
                open_positions.remove(pos)
            if closed_positions or still_pending_positions != open_positions:
                save_bot_state(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)

            # ── Step 3: Pyramiding ──────────────────────────────
            is_sl_cooldown = last_sl_time is not None and (now - last_sl_time).total_seconds() < SL_COOLDOWN_MINUTES * 60
            if (len(open_positions) < MAX_PYRAMID_POSITIONS
                    and daily_trades < MAX_TRADES_PER_DAY
                    and barrier_model is not None
                    and not is_sl_cooldown):
                for pos in open_positions:
                    if pos.get('status') != 'open' or pos['action'] != 'BUY' or pos.get('pyramided'):
                        continue
                    near_tp = current_price >= pos['take_profit'] * (1 - PYRAMID_TRIGGER_PCT)
                    if not near_tp:
                        continue

                    try:
                        df_pyr = fetch_ohlcv(SYMBOL, '5m', limit=LIVE_CANDLES)
                        df_pyr = add_features(df_pyr)
                        pyr_barrier_probs = predict_barrier_probabilities(
                            barrier_model, barrier_scaler, barrier_encoder, df_pyr
                        )
                        still_bullish, reason = should_enter(pyr_barrier_probs)
                    except Exception as e:
                        print(f"  ⚠️  Pyramid barrier check failed: {e}")
                        still_bullish = False
                        reason = "barrier check failed"

                    pos['pyramided'] = True

                    if still_bullish:
                        new_entry = round(current_price * (1 - ENTRY_OFFSET_PCT), decimals)
                        rung_tp_pct = pyr_barrier_probs.get('recommended_tp', TP_PCT)
                        new_tp    = round(new_entry * (1 + rung_tp_pct), decimals)
                        new_sl    = round(new_entry * (1 - SL_PCT), decimals)

                        new_pos = open_position_with_limit_buy(new_entry, new_tp, new_sl, now, 'pyramid entry')
                        if new_pos:
                            open_positions.append(new_pos)
                            daily_trades += 1
                            send_message(
                                f"🔼 <b>[{SYMBOL}] Pyramid entry #{len(open_positions)} (pending fill)</b>\n\n"
                                f"💰 Entry: <b>{new_entry}</b>\n"
                                f"🎯 TP:    <b>{new_tp} (+{rung_tp_pct*100:.0f}% Fib)</b>\n"
                                f"🛑 SL:    <b>{new_sl}</b>\n"
                                f"📊 {reason}"
                            )
                            save_bot_state(SYMBOL, open_positions, daily_trades, daily_pnl, last_day)
                    break

        except Exception as e:
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Position check error: {e}")

        # Check Daily Limits
        if daily_trades >= MAX_TRADES_PER_DAY:
            open_count = len(open_positions)
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Max trades reached | Open positions: {open_count}")
            if open_count > 0 and not limit_notice_sent:
                pos_summary = "\n".join([
                    f"  🟢 BUY | Entry: {p['entry']} | TP: {p['take_profit']} | SL: {p['stop_loss']} | {p.get('status')}"
                    for p in open_positions
                ])
                send_message(
                    f"⏳ <b>[{SYMBOL}] Daily trade limit reached — {open_count} position(s) still open</b>\n\n"
                    f"{pos_summary}\n\n"
                    f"💵 Current price: <b>{current_price}</b>\n"
                    f"📊 Daily PnL: <b>{daily_pnl:.2f}%</b>"
                )
                limit_notice_sent = True
            time.sleep(60)
            continue
        else:
            limit_notice_sent = False

        if daily_pnl <= -MAX_DAILY_LOSS:
            notify_loss_limit(SYMBOL)
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] Daily loss limit hit, stopping for today...")
            time.sleep(3600)
            continue

        if last_sl_time is not None:
            seconds_since_sl = (now - last_sl_time).total_seconds()
            if seconds_since_sl < SL_COOLDOWN_MINUTES * 60:
                mins_left = int((SL_COOLDOWN_MINUTES * 60 - seconds_since_sl) // 60) + 1
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] SL cooldown active ({mins_left}m remaining)")
                time.sleep(60)
                continue
            else:
                last_sl_time = None

        if last_tp_time is not None:
            seconds_since_tp = (now - last_tp_time).total_seconds()
            if seconds_since_tp < TP_COOLDOWN_MINUTES * 60:
                mins_left = int((TP_COOLDOWN_MINUTES * 60 - seconds_since_tp) // 60) + 1
                print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] TP cooldown active ({mins_left}m remaining)")
                time.sleep(60)
                continue
            else:
                last_tp_time = None

        # Fresh Signal Processing
        try:
            df     = fetch_ohlcv(SYMBOL, '5m', limit=LIVE_CANDLES)
            df     = add_features(df)
            signal = generate_signal(model, scaler, encoder, df)

            coin        = SYMBOL.split('/')[0]
            liq, levels = get_liq_data(coin, current_price)

            if signal['action'] == 'SELL':
                signal['action'] = None

            if signal['action'] and liq:
                if signal['action'] == 'BUY' and liq['bias'] == 'BEARISH':
                    signal['action'] = None
                if liq['liq_spike']:
                    signal['action'] = None

            barrier_probs = {}
            if signal['action'] == 'BUY':
                decimals = get_decimal_places(current_price)
                enter    = False

                if len(open_positions) > 0:
                    signal['action'] = None
                elif barrier_model is not None:
                    try:
                        barrier_probs = predict_barrier_probabilities(
                            barrier_model, barrier_scaler, barrier_encoder, df
                        )
                        enter, reason = should_enter(barrier_probs)
                    except Exception as e:
                        print(f"  ⚠️ Barrier prediction failed: {e}")
                        enter = False
                else:
                    print("  ⚠️ No barrier model loaded — skipping trade")

                if not enter:
                    signal['action'] = None

            if signal['action'] == 'BUY':
                model_tp = signal.get('fib_target_pct', 0.03)
                barrier_tp = barrier_probs.get('recommended_tp', 0.03) if barrier_model is not None else model_tp
                effective_tp = max(model_tp, barrier_tp)

                adjusted_entry = round(current_price * (1 - ENTRY_OFFSET_PCT), decimals)
                signal['fib_target_pct'] = effective_tp
                signal['entry']       = adjusted_entry
                signal['take_profit'] = round(adjusted_entry * (1 + effective_tp), decimals)
                signal['stop_loss']   = round(adjusted_entry * (1 - SL_PCT), decimals)

            liq_info = f"Liq: {liq['bias']} ({liq['liq_ratio']:.0%})" if liq else "Liq: N/A"
            print(f"[{SYMBOL}] [{now.strftime('%H:%M')}] "
                  f"Signal: {signal['action'] or 'NONE':4} | "
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

                new_pos = open_position_with_limit_buy(
                    signal['entry'], signal['take_profit'], signal['stop_loss'], now, 'fresh entry'
                )
                if new_pos:
                    open_positions.append(new_pos)
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
        age_hours  = (datetime.now() - metadata['trained_at']).total_seconds() / 3600
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

    print("Bot started. Press Ctrl+C to stop.")
    print("=" * 50)

    try:
        run_bot(model, scaler, encoder, barrier_model, barrier_scaler, barrier_encoder)
    except KeyboardInterrupt:
        notify_stop(SYMBOL)
        print("\nBot stopped by user")