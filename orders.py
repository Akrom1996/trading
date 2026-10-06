"""
Binance order execution via ccxt.

SAFETY MODEL — three layers:
  1. DRY_RUN=true (default)   -- logs what WOULD be ordered, places nothing.
  2. BINANCE_TESTNET=true     -- talks to Binance's Spot Testnet (fake funds).
  3. Real API key, TESTNET=false, DRY_RUN=false -- live trading.

Requires in .env:
    BINANCE_API_KEY
    BINANCE_API_SECRET
    BINANCE_TESTNET=true|false   (default: true)
    DRY_RUN=true|false          (default: true)

Install: pip install ccxt

NON-BLOCKING DESIGN NOTE:
This module does NOT poll-and-wait for a limit order to fill. A limit
order placed below market (per ENTRY_OFFSET_PCT) may take minutes or
may never fill -- blocking the bot's main loop on that would freeze
position checks, retrains, and every other symbol sharing the process
for as long as the wait takes, which could be indefinite.

Instead: place_limit_buy() returns immediately after placing the order.
check_order_status() is a single, non-blocking check of that order's
current state -- call it once per main loop tick (same ~60s cadence as
everything else in main.py), and place the OCO sell only once you see
'closed'. See main.py's pending-order handling in the position-check
block for how this is wired together.
"""

import os
import ccxt

BINANCE_API_KEY    = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BINANCE_TESTNET    = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
DRY_RUN            = os.getenv('DRY_RUN', 'true').lower() == 'true'

_exchange = None


def _get_exchange():
    global _exchange
    if _exchange is not None:
        return _exchange

    if not DRY_RUN and (not BINANCE_API_KEY or not BINANCE_API_SECRET):
        raise RuntimeError(
            "BINANCE_API_KEY / BINANCE_API_SECRET not set -- cannot place "
            "real orders. Set DRY_RUN=true if you just want to log intended "
            "orders without executing them."
        )

    _exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY or '',
        'secret': BINANCE_API_SECRET or '',
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })

    if BINANCE_TESTNET:
        _exchange.set_sandbox_mode(True)
        print("[orders] ⚠️  Running against Binance SPOT TESTNET (fake funds)")
    elif not DRY_RUN:
        print("[orders] 🔴 Running against Binance LIVE account — real funds")

    try:
        _exchange.load_markets()
    except Exception as e:
        if not DRY_RUN:
            raise RuntimeError(f"Failed to load Binance markets: {e}")

    return _exchange


def get_available_balance(asset: str) -> float:
    """Returns available free balance for a single asset (e.g., 'USDT').
    Pass the bare asset code, NOT a trading pair like 'ZEC/USDT'."""
    if DRY_RUN:
        return 10000.0  # Simulated balance
    exchange = _get_exchange()
    balance = exchange.fetch_balance()
    return float(balance.get(asset, {}).get('free', 0.0))


def place_limit_buy(symbol: str, limit_price: float, quote_amount: float) -> dict:
    """
    Places a LIMIT buy order and returns IMMEDIATELY -- does not wait
    for it to fill. Check fill status later with check_order_status().
    """
    exchange = _get_exchange()

    if not DRY_RUN:
        # A buy spends the QUOTE asset (USDT in 'ZEC/USDT'), so that is
        # the balance to check -- not the trading pair string itself.
        quote_asset = symbol.split('/')[1]
        balance = get_available_balance(quote_asset)
        if balance < quote_amount:
            raise RuntimeError(
                f"Insufficient balance: {balance:.2f} {quote_asset} available, "
                f"{quote_amount:.2f} {quote_asset} required."
            )

    formatted_price = float(exchange.price_to_precision(symbol, limit_price)) if not DRY_RUN else limit_price
    raw_amount = quote_amount / formatted_price
    formatted_amount = float(exchange.amount_to_precision(symbol, raw_amount)) if not DRY_RUN else round(raw_amount, 6)

    if DRY_RUN:
        print(f"[orders] 🧪 DRY_RUN limit buy: {formatted_amount} {symbol.split('/')[0]} @ {formatted_price} (~{quote_amount:.2f} USDT)")
        return {
            'id': 'dry-run-limit-buy',
            'symbol': symbol,
            'limit_price': formatted_price,
            'filled_amount': formatted_amount,
            'avg_price': formatted_price,
            'dry_run': True,
            'status': 'closed',  # dry run simulates an instant fill
        }

    order = exchange.create_order(
        symbol=symbol, type='limit', side='buy',
        amount=formatted_amount, price=formatted_price,
    )
    filled = float(order.get('filled', 0) or 0)
    print(f"[orders] 📝 LIVE limit buy placed: {formatted_amount} {symbol.split('/')[0]} "
          f"@ {formatted_price} (id: {order.get('id')}, status: {order.get('status')})")
    return {
        'id': order.get('id'),
        'symbol': symbol,
        'limit_price': formatted_price,
        'filled_amount': filled,
        'avg_price': formatted_price,
        'dry_run': False,
        'status': order.get('status'),
    }


def check_order_status(symbol: str, order_id: str) -> dict:
    """
    ONE non-blocking check of an order's current state. Call this once
    per main loop tick (not in a loop within this function) for any
    order you're waiting on -- never poll-and-sleep here.

    Returns: {'status': 'open'|'closed'|'canceled'|..., 'filled_amount',
    'avg_price'}. On a transient fetch error, returns
    {'status': 'unknown', ...} rather than raising, so the caller's
    normal per-tick error handling applies instead of a special case.
    """
    if order_id == 'dry-run-limit-buy':
        return {'status': 'closed', 'filled_amount': None, 'avg_price': None}

    try:
        exchange = _get_exchange()
        order_info = exchange.fetch_order(order_id, symbol=symbol)
        filled = float(order_info.get('filled', 0) or 0)
        avg_price = order_info.get('average')
        avg_price = float(avg_price) if avg_price else None
        return {
            'status': order_info.get('status'),
            'filled_amount': filled,
            'avg_price': avg_price,
        }
    except Exception as e:
        print(f"[orders] check_order_status error for {order_id}: {e}")
        return {'status': 'unknown', 'filled_amount': None, 'avg_price': None}


def cancel_order(symbol: str, order_id: str) -> bool:
    """Cancels an open order (e.g. a limit buy that's taken too long to
    fill). Returns True on success, False on failure (logged, not raised)."""
    if order_id == 'dry-run-limit-buy':
        return True
    try:
        exchange = _get_exchange()
        exchange.cancel_order(order_id, symbol=symbol)
        print(f"[orders] 🚫 Canceled unfilled order {order_id} for {symbol}")
        return True
    except Exception as e:
        print(f"[orders] cancel_order failed for {order_id}: {e}")
        return False


def place_oco_sell(symbol: str, base_amount: float, tp_price: float, sl_price: float) -> dict:
    """
    Places an OCO sell on Binance via the raw endpoint (ccxt has no
    unified 'OCO' order type). Call ONLY after the buy is confirmed filled.
    """
    exchange = _get_exchange()

    sl_limit_price = sl_price * 0.998  # buffer so the SL leg fills in a fast drop

    if DRY_RUN:
        print(f"[orders] 🧪 DRY_RUN OCO Sell set: {round(base_amount, 6)} {symbol.split('/')[0]}")
        print(f"         Target TP: {tp_price} | Target SL Trigger: {sl_price}")
        return {
            'id': 'dry-run-oco-sell', 'symbol': symbol,
            'filled_amount': round(base_amount, 6),
            'take_profit': tp_price, 'stop_loss': sl_price,
            'dry_run': True, 'status': 'open',
        }

    # Buy fees are often taken from the base coin, so the free balance can
    # be slightly below the filled amount. Never ask to sell more than we hold.
    base_asset = symbol.split('/')[0]
    free_base = get_available_balance(base_asset)
    sell_amount = min(base_amount, free_base) if free_base > 0 else base_amount

    market = exchange.market(symbol)
    qty   = exchange.amount_to_precision(symbol, sell_amount)
    tp    = exchange.price_to_precision(symbol, tp_price)
    sl_tr = exchange.price_to_precision(symbol, sl_price)
    sl_lm = exchange.price_to_precision(symbol, sl_limit_price)

    # New endpoint: POST /api/v3/orderList/oco
    new_params = {
        'symbol': market['id'],
        'side': 'SELL',
        'quantity': qty,
        'aboveType': 'LIMIT_MAKER',          # take-profit leg
        'abovePrice': tp,
        'belowType': 'STOP_LOSS_LIMIT',      # stop-loss leg
        'belowStopPrice': sl_tr,
        'belowPrice': sl_lm,
        'belowTimeInForce': 'GTC',
    }
    new_fn = (getattr(exchange, 'private_post_orderlist_oco', None)
              or getattr(exchange, 'privatePostOrderListOco', None))

    if new_fn is not None:
        order = new_fn(new_params)
    else:
        # Older ccxt without the new endpoint: legacy OCO call
        old_fn = (getattr(exchange, 'private_post_order_oco', None)
                  or getattr(exchange, 'privatePostOrderOco'))
        order = old_fn({
            'symbol': market['id'], 'side': 'SELL', 'quantity': qty,
            'price': tp, 'stopPrice': sl_tr, 'stopLimitPrice': sl_lm,
            'stopLimitTimeInForce': 'GTC',
        })

    print(f"[orders] 🎯 LIVE OCO Sell set for {symbol}: TP @ {tp} | SL @ {sl_tr} "
          f"(orderListId: {order.get('orderListId')})")
    return order

def place_market_buy(symbol: str, quote_amount: float, tp_price: float = None, sl_price: float = None) -> dict:
    """Market buy for instant fills (no offset, no wait)."""
    exchange = _get_exchange()

    if DRY_RUN:
        try:
            ticker = exchange.fetch_ticker(symbol)
            avg_price = float(ticker['last'])
        except Exception:
            avg_price = 100.0
        est_filled = round(quote_amount / avg_price, 6)
        print(f"[orders] 🧪 DRY_RUN market buy: {quote_amount} USDT of {symbol} (~{est_filled} @ {avg_price:.4f})")
        return {
            'id': 'dry-run-market-buy', 'symbol': symbol,
            'filled_amount': est_filled, 'avg_price': avg_price,
            'take_profit': tp_price, 'stop_loss': sl_price,
            'dry_run': True, 'status': 'closed',
        }

    formatted_quote_amount = exchange.cost_to_precision(symbol, quote_amount)
    order = exchange.create_order(
        symbol=symbol, type='market', side='buy',
        amount=None, params={'quoteOrderQty': formatted_quote_amount},
    )
    filled = float(order.get('filled', 0) or 0)
    cost   = float(order.get('cost', 0) or 0)
    avg_price = (cost / filled) if filled > 0 else float(order.get('price', 0) or 0)
    print(f"[orders] ✅ LIVE market buy filled: {filled} {symbol.split('/')[0]} @ avg {avg_price:.4f}")
    return {
        'id': order.get('id'), 'symbol': symbol,
        'filled_amount': filled, 'avg_price': avg_price,
        'take_profit': tp_price, 'stop_loss': sl_price,
        'dry_run': False, 'status': order.get('status'),
    }


def place_market_sell(symbol: str, base_amount: float) -> dict:
    """Direct market sell for immediate closing of positions."""
    exchange = _get_exchange()
    formatted_amount = float(exchange.amount_to_precision(symbol, base_amount)) if not DRY_RUN else round(base_amount, 6)

    if DRY_RUN:
        try:
            ticker = exchange.fetch_ticker(symbol)
            avg_price = float(ticker['last'])
        except Exception:
            avg_price = 100.0
        print(f"[orders] 🧪 DRY_RUN market sell: {formatted_amount} {symbol.split('/')[0]}")
        return {
            'id': 'dry-run-market-sell', 'symbol': symbol,
            'filled_amount': formatted_amount, 'avg_price': avg_price,
            'dry_run': True, 'status': 'closed',
        }

    order = exchange.create_order(
        symbol=symbol, type='market', side='sell', amount=formatted_amount,
    )
    filled = float(order.get('filled', 0) or 0)
    cost   = float(order.get('cost', 0) or 0)
    avg_price = (cost / filled) if filled > 0 else float(order.get('price', 0) or 0)
    print(f"[orders] ✅ LIVE market sell filled: {filled} {symbol.split('/')[0]} @ avg {avg_price:.4f}")
    return {
        'id': order.get('id'), 'symbol': symbol,
        'filled_amount': filled, 'avg_price': avg_price,
        'dry_run': False, 'status': order.get('status'),
    }