"""
Altcoin Signal Evaluation & Execution Engine.
Manages symbol lifecycle, technical indicators, ML & Fibonacci barrier predictions,
pending limit-buy monitoring, OCO safety, pyramiding, and trade journaling.
"""

import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from config.settings import settings
from src.exchanges.binance_client import (
    check_oco,
    fetch_ohlcv,
    fetch_current_price,
    place_limit_buy,
    place_market_buy,
    place_market_sell,
    place_oco_sell,
    check_order_status,
    cancel_order,
)
from src.indicators.technicals import add_features, add_labels
from src.indicators.heatmap import get_liq_data
from src.services.redis_cache import (
    init_db,
    save_bot_state,
    load_bot_state,
    record_closed_trade,
    get_last_sl_time,
    get_last_tp_time,
    get_shared_value,
)
from src.services.telegram_notifier import (
    notify_signal_with_liq,
    notify_error,
    notify_loss_limit,
    notify_start,
    notify_stop,
    send_message,
)
from src.indicators.technicals import (
    add_features,
    add_labels,
    train_model,
    save_model,
    load_model,
    model_is_fresh,
    train_barrier_model,
    save_barrier_model,
    load_barrier_model,
    predict_barrier_probabilities,
    should_enter,
    generate_signal,
    TP_PCT,
    SL_PCT,
    FEATURES,
)

VALID_STATUSES = ('open', 'pending_fill')


def get_decimal_places(price: float) -> int:
    if price >= 100:  return 2
    if price >= 1:    return 4
    if price >= 0.01: return 6
    return 8


def normalize_position(pos: dict, decimals: int, symbol: str) -> dict:
    if 'take_profit' not in pos:
        pos['take_profit'] = round(pos['entry'] * (1 + TP_PCT), decimals)
    if 'stop_loss' not in pos:
        pos['stop_loss'] = round(pos['entry'] * (1 - settings.MAX_RISK_PER_TRADE), decimals)
    if 'pyramided' not in pos:
        pos['pyramided'] = True
    if 'status' not in pos:
        pos['status'] = 'open'
    if pos.get('status') not in VALID_STATUSES:
        print(f"[{symbol}] Migrating legacy position status '{pos.get('status')}' -> 'open' (virtual)")
        pos['status'] = 'open'
        pos['virtual'] = True
    if 'virtual' not in pos:
        pos['virtual'] = False
    pos.pop('peak_price', None)
    pos.pop('trailing_stop', None)
    return pos


def close_position(pos: dict, current_price: float, reason: str, daily_pnl: float, symbol: str) -> float:
    if pos["action"] == "BUY":
        pnl = ((current_price - pos["entry"]) / pos["entry"]) * 100
    else:
        pnl = ((pos["entry"] - current_price) / pos["entry"]) * 100

    icon = "✅" if pnl >= 0 else "🛑"
    virtual_tag = " 🔸<i>(signal only — not filled on this account)</i>" if pos.get('virtual') else ""
    msg = (
        f"{icon} <b>[{symbol}] {pos['action']} Closed — {reason}</b>{virtual_tag}\n\n"
        f"💰 Entry:  <b>{pos['entry']}</b>\n"
        f"🎯 Close:  <b>{current_price}</b>\n"
        f"📈 PnL:    <b>{pnl:+.2f}%</b>"
    )
    send_message(msg)

    if not pos.get('virtual') and not pos.get('oco_order_id') and pos.get('base_amount'):
        try:
            place_market_sell(symbol, pos['base_amount'])
        except Exception as e:
            notify_error(f"[{symbol}] SELL order failed on close: {e}")
            print(f"[{symbol}] Order error on close: {e}")

    record_closed_trade(
        symbol=symbol, action=pos["action"], entry=pos["entry"],
        exit_price=current_price, pnl_pct=pnl, reason=reason,
    )
    return daily_pnl + pnl


def process_pending_fill(pos: dict, now: datetime, symbol: str) -> str:
    info = check_order_status(symbol, pos['buy_order_id'])
    status = info['status']

    if status == 'closed':
        apply_fill(pos, info['filled_amount'], info['avg_price'])
        return 'filled'

    if status == 'canceled':
        if info['filled_amount']:                       # partially filled, then canceled
            apply_fill(pos, info['filled_amount'], info['avg_price'])
            return 'filled'
        return 'gave_up'

    if pos.get('order_type') == 'market':               # market orders never time out
        return 'still_pending'

    placed_at = pos.get('order_placed_at')
    if isinstance(placed_at, str):
        try:
            placed_at = datetime.fromisoformat(placed_at)
        except ValueError:
            placed_at = None

    if placed_at and (now - placed_at).total_seconds() > settings.PENDING_FILL_TIMEOUT_MINUTES * 60:
        if not cancel_order(symbol, pos['buy_order_id']):
            return 'still_pending'                      # cancel failed, try again next loop
        info = check_order_status(symbol, pos['buy_order_id'])
        if info['filled_amount']:                       # partial fill before cancel
            apply_fill(pos, info['filled_amount'], info['avg_price'])
            return 'filled'
        return 'gave_up'

    return 'still_pending'


def apply_fill(pos: dict, filled, avg_price):
    pos['status'] = 'open'
    pos['base_amount'] = filled or pos.get('base_amount') or 0.0
    if avg_price:
        pos['entry'] = avg_price
        if pos.get('order_type') == 'market':      # rebase TP/SL on the real fill
            d = get_decimal_places(avg_price)
            pos['take_profit'] = round(avg_price * (1 + pos['tp_pct']), d)
            pos['stop_loss'] = round(avg_price * (1 - pos['sl_pct']), d)


def protect_position(pos: dict, symbol: str):
    try:
        oco = place_oco_sell(symbol, pos['base_amount'], pos['take_profit'], pos['stop_loss'])
        pos['oco_order_id'] = oco.get('id') or oco.get('orderListId')
    except Exception as e:
        notify_error(f"🚨 [{symbol}] UNPROTECTED POSITION — buy filled but OCO failed: {e}")


def open_position(price: float, tp_pct: float, sl_pct: float, now: datetime,
                  label: str, symbol: str, is_pyramid: bool = False,
                  order_type: Optional[str] = None) -> dict:
    if order_type is None:
        order_type = 'limit' if is_pyramid else 'market'
    d = get_decimal_places(price)
    pos = {
        'action': 'BUY',
        'entry': price,
        'take_profit': round(price * (1 + tp_pct), d),
        'stop_loss': round(price * (1 - sl_pct), d),
        'tp_pct': tp_pct,
        'sl_pct': sl_pct,
        'order_type': order_type,
        'pyramided': False,                 # parent gets set to True in Step 3
        'status': 'pending_fill',
        'buy_order_id': None,
        'order_placed_at': now.isoformat(),
        'base_amount': 0.0,
        'oco_order_id': None,
        'virtual': False,
        'opened_at': now.strftime('%H:%M'),
    }
    try:
        if order_type == 'limit':
            res = place_limit_buy(symbol, price, settings.QUOTE_AMOUNT_PER_TRADE)
            pos['buy_order_id'] = res.get('id')
            pos['base_amount'] = res.get('filled_amount') or 0.0
        else:
            res = place_market_buy(symbol, settings.QUOTE_AMOUNT_PER_TRADE)
            pos['buy_order_id'] = res.get('id')
            if res.get('status') == 'closed' and res.get('filled_amount'):
                apply_fill(pos, res['filled_amount'], res['avg_price'])
                protect_position(pos, symbol)      # protected immediately, no 60s gap
            # otherwise stays pending_fill and Step 1 handles it
        return pos
    except Exception as e:
        pos.update(status='open', virtual=True, error=str(e))
        notify_error(f"⚠️ [{symbol}] {label} order failed ({e}) — tracking signal anyway")
        return pos


def retrain_and_save(symbol: str):
    print(f"[{symbol}] Fetching {settings.TRAIN_CANDLES} candles...")
    df = fetch_ohlcv(symbol, '5m', limit=settings.TRAIN_CANDLES)
    df = add_features(df)
    df = add_labels(df)
    print(f"  Candles loaded : {len(df)}")
    print("Training model...")
    model, scaler, encoder = train_model(df)
    save_model(model, scaler, encoder, symbol=symbol, candles_count=len(df))
    return model, scaler, encoder


def retrain_barrier_and_save(symbol: str):
    df_barrier = fetch_ohlcv(symbol, '5m', limit=settings.TRAIN_CANDLES)
    df_barrier = add_features(df_barrier)
    barrier_model, barrier_scaler, barrier_encoder, label_stats = train_barrier_model(df_barrier, symbol=symbol)
    save_barrier_model(barrier_model, barrier_scaler, barrier_encoder, symbol=symbol, candles_count=len(df_barrier), label_stats=label_stats)

    fib_tp_pct = sum(stats.get('pct', 0) for name, stats in label_stats.items() if 'TP_FIB' in name)
    if fib_tp_pct < 5:
        send_message(
            f"⚠️ <b>[{symbol}] Barrier model imbalance warning</b>\n"
            f"Profitable Fibonacci examples are only {fib_tp_pct:.1f}% of training data — live probabilities may stay low."
        )
    return barrier_model, barrier_scaler, barrier_encoder


def run_coin_trader(symbol: Optional[str] = None):
    symbol = symbol or settings.SYMBOL
    print("=" * 50)
    print(f"  {symbol} Coin Trader Initializing")
    print("=" * 50)

    init_db()
    notify_start(symbol)

    # 1. Load or train directional model
    print(f"[{symbol}] Checking for saved model...")
    model, scaler, encoder, metadata = load_model(symbol=symbol)
    if model is not None and model_is_fresh(metadata, max_age_hours=settings.MAX_MODEL_AGE_HRS):
        trained_at = metadata['trained_at'].strftime('%Y-%m-%d %H:%M')
        age_hours = (datetime.now() - metadata['trained_at']).total_seconds() / 3600
        print(f"  Saved model found! Trained at {trained_at} ({age_hours:.1f}h old)")
    else:
        print(f"[{symbol}] Training fresh directional model...")
        model, scaler, encoder = retrain_and_save(symbol)

    # 2. Load or train barrier model
    print(f"[{symbol}] Checking for saved barrier model...")
    barrier_model, barrier_scaler, barrier_encoder, barrier_meta = load_barrier_model(symbol=symbol)
    if barrier_model is None:
        print(f"[{symbol}] Training fresh barrier model...")
        barrier_model, barrier_scaler, barrier_encoder = retrain_barrier_and_save(symbol)
    else:
        print(f"  Saved barrier model found (trained {barrier_meta['trained_at']})")

    # 3. Restore state from DB/Redis
    saved_positions, saved_trades, saved_pnl, saved_day = load_bot_state(symbol)
    now0 = datetime.now()
    if saved_positions or saved_day == now0.day:
        open_positions = saved_positions or []
        daily_trades = saved_trades if saved_day == now0.day else 0
        daily_pnl = saved_pnl if saved_day == now0.day else 0.0
        if open_positions:
            try:
                _price_for_decimals = fetch_current_price(symbol)
                _decimals = get_decimal_places(_price_for_decimals)
            except Exception:
                _decimals = 4
            open_positions = [normalize_position(p, _decimals, symbol) for p in open_positions]
        print(f"[{symbol}] Restored {len(open_positions)} open position(s) from DB")
        if open_positions:
            send_message(f"🔄 <b>[{symbol}] Restarted — restored {len(open_positions)} open position(s) from DB</b>")
            save_bot_state(symbol, open_positions, daily_trades, daily_pnl, now0.day)
    else:
        open_positions = []
        daily_trades = 0
        daily_pnl = 0.0

    last_retrain = datetime.now()
    last_day = datetime.now().day
    limit_notice_sent = False

    last_sl_time = get_last_sl_time(symbol)
    if last_sl_time is not None:
        secs_since_sl = (now0 - last_sl_time).total_seconds()
        if secs_since_sl < settings.SL_COOLDOWN_MINUTES * 60:
            mins_left = int((settings.SL_COOLDOWN_MINUTES * 60 - secs_since_sl) // 60) + 1
            print(f"[{symbol}] Active Stop Loss cooldown restored: {mins_left}m remaining")
        else:
            last_sl_time = None

    last_tp_time = get_last_tp_time(symbol)
    if last_tp_time is not None:
        secs_since_tp = (now0 - last_tp_time).total_seconds()
        if secs_since_tp < settings.TP_COOLDOWN_MINUTES * 60:
            mins_left = int((settings.TP_COOLDOWN_MINUTES * 60 - secs_since_tp) // 60) + 1
            print(f"[{symbol}] Active TP cooldown restored: {mins_left}m remaining")
        else:
            last_tp_time = None

    print(f"[{symbol}] 🚀 Trading loop active. Polling market...")

    # 4. Main runtime loop
    while True:
        now = datetime.now()

        # Midnight reset
        if now.day != last_day:
            daily_trades = 0
            daily_pnl = 0.0
            last_day = now.day
            limit_notice_sent = False
            print(f"[{symbol}] [{now.strftime('%H:%M')}] Daily counters reset (holding {len(open_positions)} positions)")
            holding_msg = f"\nHolding <b>{len(open_positions)}</b> open position(s) into the new day." if open_positions else ""
            send_message(f"🔄 <b>[{symbol}] Daily counters reset</b>{holding_msg}")
            save_bot_state(symbol, open_positions, daily_trades, daily_pnl, last_day)

        # Periodic Retrain
        if (now - last_retrain).total_seconds() / 60 >= 60:
            print(f"[{symbol}] [{now.strftime('%H:%M')}] Retraining models...")
            try:
                model, scaler, encoder = retrain_and_save(symbol)
                last_retrain = now
            except Exception as e:
                notify_error(f"[{symbol}] Retrain failed: {e}")

            try:
                barrier_model, barrier_scaler, barrier_encoder = retrain_barrier_and_save(symbol)
            except Exception as e:
                notify_error(f"[{symbol}] Barrier retrain failed: {e}")

        try:
            df_recent = fetch_ohlcv(symbol, '5m', limit=2)
            latest_candle = df_recent.iloc[-1]
            candle_high = float(latest_candle['high'])
            candle_low = float(latest_candle['low'])
            current_price = float(latest_candle['close'])
            decimals = get_decimal_places(current_price)

            # Step 1: Advance pending limit-buy orders
            state_changed = False
            still_pending = []
            for pos in open_positions:
                if pos.get('status') != 'pending_fill':
                    still_pending.append(pos)
                    continue

                outcome = process_pending_fill(pos, now, symbol)
                if outcome == 'still_pending':
                    still_pending.append(pos)
                elif outcome == 'gave_up':
                    print(f"[{symbol}] Limit buy for position opened at {pos.get('opened_at')} canceled (unfilled)")
                    send_message(f"⏱️ <b>[{symbol}] Limit buy never filled, canceled</b>\nTarget entry: {pos['entry']}")
                    daily_trades = max(0, daily_trades - 1)
                    state_changed = True
                elif outcome == 'filled':
                    print(f"[{symbol}] ✅ Limit buy filled @ {pos['entry']} — placing OCO")
                    state_changed = True
                    if not pos.get('oco_order_id'):
                        protect_position(pos, symbol)
                    still_pending.append(pos)

            open_positions = still_pending

            # Step 2: Candle-based TP / SL check
            closed_positions = []
            for pos in open_positions:
                if pos.get('status') != 'open' or pos['action'] != 'BUY':
                    continue

                oco_id = pos.get('oco_order_id')

                # Real OCO: Binance is the source of truth
                if oco_id is not None and not pos.get('virtual'):
                    r = check_oco(symbol, oco_id, pos['take_profit'], pos['stop_loss'])
                    state = r['state']

                    if state == 'tp':
                        daily_pnl = close_position(pos, r['exit_price'] or pos['take_profit'],
                                                "TP Hit (OCO filled)", daily_pnl, symbol)
                        closed_positions.append(pos); last_tp_time = now
                    elif state == 'sl':
                        daily_pnl = close_position(pos, r['exit_price'] or pos['stop_loss'],
                                                "SL Hit (OCO filled)", daily_pnl, symbol)
                        closed_positions.append(pos); last_sl_time = now
                    elif state in ('canceled', 'rejected'):
                        # Coins are still held but unprotected: re-arm
                        notify_error(f"⚠️ [{symbol}] OCO {oco_id} ended without fill ({state}) — re-placing")
                        pos['oco_order_id'] = None
                        protect_position(pos, symbol)
                    # 'open' / 'unknown' -> do nothing this loop
                    continue

                if candle_high >= pos['take_profit']:
                    daily_pnl = close_position(pos, pos['take_profit'], "TP Hit (Candle High)", daily_pnl, symbol)
                    closed_positions.append(pos)
                    last_tp_time = now
                    print(f"⏸️ [{symbol}] TP Cooldown Initiated ({settings.TP_COOLDOWN_MINUTES}m)")

                elif candle_low <= pos['stop_loss']:
                    daily_pnl = close_position(pos, pos['stop_loss'], "SL Hit (Candle Low)", daily_pnl, symbol)
                    closed_positions.append(pos)
                    last_sl_time = now
                    print(f"⏸️ [{symbol}] Stop Loss Cooldown Initiated ({settings.SL_COOLDOWN_MINUTES}m)")

            for pos in closed_positions:
                open_positions.remove(pos)
            if closed_positions or state_changed:
                save_bot_state(symbol, open_positions, daily_trades, daily_pnl, last_day)

            # Step 3: Pyramiding
            is_sl_cooldown = last_sl_time is not None and (now - last_sl_time).total_seconds() < settings.SL_COOLDOWN_MINUTES * 60
            if (len(open_positions) < settings.MAX_PYRAMID_POSITIONS
                    and daily_trades < settings.MAX_TRADES_PER_DAY
                    and barrier_model is not None
                    and not is_sl_cooldown):
                for pos in open_positions:
                    if pos.get('status') != 'open' or pos['action'] != 'BUY' or pos.get('pyramided'):
                        continue
                    near_tp = current_price >= pos['take_profit'] * (1 - settings.PYRAMID_TRIGGER_PCT)
                    if not near_tp:
                        continue

                    try:
                        df_pyr = fetch_ohlcv(symbol, '5m', limit=settings.LIVE_CANDLES)
                        df_pyr = add_features(df_pyr)
                        pyr_barrier_probs = predict_barrier_probabilities(barrier_model, barrier_scaler, barrier_encoder, df_pyr)
                        still_bullish, reason = should_enter(pyr_barrier_probs)
                    except Exception as e:
                        still_bullish = False
                        reason = f"barrier check failed: {e}"

                    pos['pyramided'] = True

                    if still_bullish:
                        new_entry = round(current_price * (1 - settings.ENTRY_OFFSET_PCT), decimals)
                        rung_tp_pct = pyr_barrier_probs.get('recommended_tp', TP_PCT)
                        new_pos = open_position(new_entry, rung_tp_pct, settings.MAX_RISK_PER_TRADE, now, 'pyramid entry', symbol, is_pyramid=True)
                        open_positions.append(new_pos)

                        daily_trades += 1
                        virtual_tag = " 🔸(signal only)" if new_pos.get('virtual') else " (pending fill)"
                        send_message(
                            f"🔼 <b>[{symbol}] Pyramid entry #{len(open_positions)}{virtual_tag}</b>\n\n"
                            f"💰 Entry: <b>{new_entry}</b>\n"
                            f"🎯 TP:    <b>{new_pos['take_profit']} (+{rung_tp_pct*100:.1f}% Fib)</b>\n"
                            f"🛑 SL:    <b>{new_pos['stop_loss']}</b>\n"
                            f"📊 {reason}"
                        )
                    save_bot_state(symbol, open_positions, daily_trades, daily_pnl, last_day)
                    break

        except Exception as e:
            print(f"[{symbol}] [{now.strftime('%H:%M')}] Position loop check error: {e}")

        # Step 4: Daily Limits
        if daily_trades >= settings.MAX_TRADES_PER_DAY:
            open_count = len(open_positions)
            print(f"[{symbol}] [{now.strftime('%H:%M')}] Max trades reached | Open: {open_count}")
            if open_count > 0 and not limit_notice_sent:
                pos_summary = "\n".join([
                    f"  🟢 BUY | Entry: {p['entry']} | TP: {p['take_profit']} | SL: {p['stop_loss']} | {p.get('status')}"
                    for p in open_positions
                ])
                send_message(
                    f"⏳ <b>[{symbol}] Daily trade limit reached — {open_count} position(s) still open</b>\n\n"
                    f"{pos_summary}\n\n"
                    f"💵 Current price: <b>{current_price}</b>\n"
                    f"📊 Daily PnL: <b>{daily_pnl:.2f}%</b>"
                )
                limit_notice_sent = True
            time.sleep(60)
            continue
        else:
            limit_notice_sent = False

        if daily_pnl <= -settings.MAX_DAILY_LOSS:
            notify_loss_limit(symbol)
            print(f"[{symbol}] Daily loss limit hit, stopping for today...")
            time.sleep(3600)
            continue

        if last_sl_time is not None:
            seconds_since_sl = (now - last_sl_time).total_seconds()
            if seconds_since_sl < settings.SL_COOLDOWN_MINUTES * 60:
                mins_left = int((settings.SL_COOLDOWN_MINUTES * 60 - seconds_since_sl) // 60) + 1
                print(f"[{symbol}] [{now.strftime('%H:%M')}] SL cooldown active ({mins_left}m remaining)")
                time.sleep(60)
                continue
            else:
                last_sl_time = None

        if last_tp_time is not None:
            seconds_since_tp = (now - last_tp_time).total_seconds()
            if seconds_since_tp < settings.TP_COOLDOWN_MINUTES * 60:
                mins_left = int((settings.TP_COOLDOWN_MINUTES * 60 - seconds_since_tp) // 60) + 1
                print(f"[{symbol}] [{now.strftime('%H:%M')}] TP cooldown active ({mins_left}m remaining)")
                time.sleep(60)
                continue
            else:
                last_tp_time = None

        # Step 5: Fresh Signal Evaluation
        try:
            df = fetch_ohlcv(symbol, '5m', limit=settings.LIVE_CANDLES)
            df = add_features(df)
            signal = generate_signal(model, scaler, encoder, df)

            # -------------------------------------------------------------
            # STEP 1: CHECK BTC MACRO REGIME FIRST (Circuit Breaker)
            # -------------------------------------------------------------
            btc_macro = get_shared_value("btc:macro_regime")
            btc_bias = (btc_macro.get('bias', 'NEUTRAL') if btc_macro else 'NEUTRAL').strip().upper()

            # If BTC is BEARISH and signal is BUY, demote immediately and skip!
            if btc_bias == 'BEARISH' and signal.get('action') == 'BUY':
                print(f"[{symbol}] [{now.strftime('%H:%M')}] BTC macro is BEARISH. Skipping {symbol} BUY signal.")
                signal['action'] = None

            # Ignore SELL signals
            if signal.get('action') == 'SELL':
                signal['action'] = None

            # -------------------------------------------------------------
            # STEP 2: CHECK LIQUIDATIONS & BARRIER MODEL (Only if still BUY)
            # -------------------------------------------------------------
            coin = symbol.split('/')[0]
            liq, levels = get_liq_data(coin, current_price)

            if signal.get('action') == 'BUY' and liq:
                if liq['bias'] == 'BEARISH' or liq['liq_spike']:
                    signal['action'] = None

            barrier_probs = {}
            if signal.get('action') == 'BUY':
                decimals = get_decimal_places(current_price)
                enter = False

                if len(open_positions) > 0:
                    signal['action'] = None
                elif barrier_model is not None:
                    try:
                        barrier_probs = predict_barrier_probabilities(barrier_model, barrier_scaler, barrier_encoder, df)
                        enter, reason = should_enter(barrier_probs)
                    except Exception as e:
                        print(f"  ⚠️ Barrier prediction error: {e}")
                        enter = False
                else:
                    enter = False

                if not enter:
                    signal['action'] = None

            # -------------------------------------------------------------
            # STEP 3: CALCULATE ENTRY / TP / SL & EXECUTE ORDER
            # -------------------------------------------------------------
            if signal.get('action') == 'BUY':
                model_tp = signal.get('fib_target_pct', 0.03)
                barrier_tp = barrier_probs.get('recommended_tp', 0.03) if barrier_model is not None else model_tp
                effective_tp = max(model_tp, barrier_tp)

                if btc_bias == 'BULLISH':
                    entry_price = current_price
                    order_type = 'market'
                else:
                    entry_price = round(current_price * (1 - settings.ENTRY_OFFSET_PCT), decimals)
                    order_type = 'limit'

                signal['entry'] = entry_price
                signal['take_profit'] = round(entry_price * (1 + effective_tp), decimals)
                signal['stop_loss'] = round(entry_price * (1 - settings.MAX_RISK_PER_TRADE), decimals)
                signal['order_type'] = order_type

            liq_info = f"Liq: {liq['bias']} ({liq['liq_ratio']:.0%})" if liq else "Liq: N/A"
            btc_info = f"BTC: {btc_bias}"
            print(
                f"[{symbol}] [{now.strftime('%H:%M')}] "
                f"Signal: {signal['action'] or 'NONE':4} | "
                f"Price: {current_price} | "
                f"{btc_info} | "
                f"{liq_info} | "
                f"Trades: {daily_trades}/{settings.MAX_TRADES_PER_DAY}"
            )

            if signal.get('action') == 'BUY':
                level_text = ""
                if levels:
                    level_text = "\n\n🗺 <b>Nearby Liq Levels:</b>\n"
                    for lv in levels[:3]:
                        level_text += (f"  {'⬆️' if lv['direction'] == 'ABOVE' else '⬇️'} "
                                       f"${lv['price']} ({lv['distance']}% away — ${lv['amount']:,.0f})\n")

                notify_signal_with_liq(signal, daily_trades + 1, liq, level_text, symbol=symbol)

                order_label = f"fresh {order_type} entry"
                new_pos = open_position(
                    entry_price, effective_tp, settings.MAX_RISK_PER_TRADE, now,
                    order_label, symbol, order_type=order_type
                )

                open_positions.append(new_pos)
                daily_trades += 1
                save_bot_state(symbol, open_positions, daily_trades, daily_pnl, last_day)
        except Exception as e:
            notify_error(f"[{symbol}] {e}")
            print(f"[{symbol}] [{now.strftime('%H:%M')}] Error: {e}")

        time.sleep(60)

