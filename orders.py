"""
Binance order execution via ccxt.

SAFETY MODEL — three layers, check them in this order before ever
pointing this at real funds:

  1. DRY_RUN=true (default)   -- logs what WOULD be ordered, places
                                  nothing. Use this to sanity-check
                                  sizing/timing against live signals
                                  before any money is at risk.
  2. BINANCE_TESTNET=true     -- talks to Binance's Spot Testnet
                                  (fake funds, same API surface). Use
                                  this once DRY_RUN output looks right,
                                  to test the actual order-placement
                                  path without risking real capital.
  3. Real API key, TESTNET=false, DRY_RUN=false
                                -- live trading. Only flip this after
                                  1 and 2 have run clean for a while.

Requires in .env:
    BINANCE_API_KEY
    BINANCE_API_SECRET
    BINANCE_TESTNET=true|false   (default: true -- safest default)
    DRY_RUN=true|false           (default: true -- safest default)

Install: pip install ccxt
"""

import os
import ccxt

BINANCE_API_KEY    = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BINANCE_TESTNET     = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
DRY_RUN              = os.getenv('DRY_RUN', 'true').lower() == 'true'

_exchange = None
_markets_loaded = False


def _get_exchange():
    global _exchange, _markets_loaded
    if _exchange is not None:
        return _exchange

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise RuntimeError(
            "BINANCE_API_KEY / BINANCE_API_SECRET not set -- cannot place "
            "real orders. Set DRY_RUN=true if you just want to log intended "
            "orders without executing them."
        )

    _exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    if BINANCE_TESTNET:
        _exchange.set_sandbox_mode(True)
        print("[orders] ⚠️  Running against Binance SPOT TESTNET (fake funds)")
    else:
        print("[orders] 🔴 Running against Binance LIVE account — real funds")

    return _exchange


def _check_min_notional(symbol: str, quote_amount: float):
    """
    Binance rejects orders below a symbol-specific minimum notional
    value (varies per pair, commonly ~$5-10 but check per symbol).
    Fail loudly and BEFORE sending the order, rather than finding out
    from a confusing exchange error mid-trade.
    """
    if DRY_RUN:
        return  # nothing to validate against a live orderbook in dry-run
    exchange = _get_exchange()
    global _markets_loaded
    if not _markets_loaded:
        exchange.load_markets()
        _markets_loaded = True
    market = exchange.market(symbol)
    min_notional = None
    # ccxt normalizes this differently across versions/exchanges --
    # check the common locations.
    limits = market.get('limits', {})
    if 'cost' in limits and limits['cost'].get('min') is not None:
        min_notional = limits['cost']['min']
    if min_notional and quote_amount < min_notional:
        raise ValueError(
            f"QUOTE_AMOUNT_PER_TRADE={quote_amount} is below {symbol}'s "
            f"minimum order value (${min_notional}) -- order would be "
            f"rejected by Binance. Raise QUOTE_AMOUNT_PER_TRADE."
        )


def get_available_balance(asset: str) -> float:
    """e.g. get_available_balance('USDT') -> 500.0"""
    if DRY_RUN:
        return float('inf')  # dry run never blocks on balance
    exchange = _get_exchange()
    balance = exchange.fetch_balance()
    return balance.get(asset, {}).get('free', 0.0)


def place_limit_buy(symbol: str, limit_price: float, quote_amount: float,
                     take_profit: float = None, stop_loss: float = None):
    """
    Places a LIMIT buy at `limit_price` (intentionally below current
    market, per an entry offset -- see ENTRY_OFFSET_PCT in main.py)
    rather than buying immediately at market. Used when a coin's price
    tends to dip briefly after a BUY signal before the real move starts
    -- this tries to catch that dip instead of paying the pre-dip price.

    `take_profit`/`stop_loss` are NOT sent to Binance -- same as
    place_market_buy, they're accepted purely so the full trade plan
    (entry, TP, SL, size) is visible together in the log line.

    NOTE: unlike a market order, this does NOT guarantee an immediate
    fill. If price never comes back down to limit_price, the order
    sits open (or, if never filled by the time you check, effectively
    misses the trade). Returns immediately after placing -- filled_amount
    will be 0 if the order hasn't filled yet at the time of this call.
    Callers should treat a non-full fill as "position not fully open
    yet" rather than assuming the whole size is on immediately.
    """
    base_amount_estimate = round(quote_amount / limit_price, 6)
    plan = f"TP={take_profit} SL={stop_loss} size={quote_amount} USDT"

    if DRY_RUN:
        print(f"[orders] 🧪 DRY_RUN limit buy: {base_amount_estimate} {symbol.split('/')[0]} "
              f"@ {limit_price} | {plan} (no real order placed)")
        return {
            'id': 'dry-run', 'symbol': symbol, 'limit_price': limit_price,
            'filled_amount': base_amount_estimate, 'avg_price': limit_price,
            'dry_run': True, 'status': 'closed',
            'take_profit': take_profit, 'stop_loss': stop_loss,
        }

    exchange = _get_exchange()
    order = exchange.create_order(
        symbol=symbol, type='limit', side='buy',
        amount=base_amount_estimate, price=limit_price,
    )
    filled = float(order.get('filled', 0) or 0)
    print(f"[orders] 📝 LIVE limit buy placed: {base_amount_estimate} {symbol.split('/')[0]} "
          f"@ {limit_price} (order id {order.get('id')}, filled so far: {filled}, "
          f"status: {order.get('status')}) | plan: {plan}")
    return {
        'id': order.get('id'), 'symbol': symbol, 'limit_price': limit_price,
        'filled_amount': filled, 'avg_price': limit_price,
        'dry_run': False, 'status': order.get('status'),
        'take_profit': take_profit, 'stop_loss': stop_loss,
    }


def place_market_buy(symbol: str, quote_amount: float, expected_price: float = None,
                      take_profit: float = None, stop_loss: float = None):
    """
    Buys `quote_amount` worth of `symbol`'s quote currency (e.g.
    quote_amount=50 on 'ZEC/USDT' spends 50 USDT buying ZEC at market).

    `expected_price`, `take_profit`, `stop_loss` are NOT sent to Binance
    (a market order has no price/TP/SL fields) -- they're accepted here
    purely so the caller's full trade plan is visible in one place: the
    order log line, the Telegram message, and the returned dict all show
    what was intended (expected entry, TP, SL) next to what actually
    happened (real fill price, size). This makes it easy to spot slippage
    -- comparing expected_price to the real avg_price -- without cross-
    referencing two separate log lines.

    Returns a dict with at least: {'id', 'filled_amount', 'avg_price',
    'symbol', 'expected_price', 'take_profit', 'stop_loss'}. In DRY_RUN
    mode, no real order is placed -- returns a simulated fill using
    expected_price (or fails clearly if none given) so calling code can
    proceed identically either way.
    """
    plan = (f"entry~{expected_price} TP={take_profit} SL={stop_loss} "
            f"size={quote_amount} USDT")

    if DRY_RUN:
        sim_price = expected_price
        sim_filled = round(quote_amount / sim_price, 6) if sim_price else None
        print(f"[orders] 🧪 DRY_RUN buy: {symbol} | {plan} (no real order placed)")
        return {
            'id': 'dry-run', 'symbol': symbol,
            'filled_amount': sim_filled, 'avg_price': sim_price, 'dry_run': True,
            'expected_price': expected_price, 'take_profit': take_profit, 'stop_loss': stop_loss,
        }

    _check_min_notional(symbol, quote_amount)
    exchange = _get_exchange()
    # createMarketBuyOrder on Binance spot uses quoteOrderQty semantics
    # via ccxt's `params` when using 'cost' — ccxt normalizes this per
    # exchange; for Binance specifically, pass quote amount via params.
    order = exchange.create_order(
        symbol=symbol, type='market', side='buy',
        amount=None,
        params={'quoteOrderQty': quote_amount},
    )
    filled = float(order.get('filled', 0) or 0)
    cost   = float(order.get('cost', 0) or 0)
    avg_price = (cost / filled) if filled else None

    slippage_note = ""
    if expected_price and avg_price:
        slippage_pct = (avg_price - expected_price) / expected_price * 100
        slippage_note = f" | slippage: {slippage_pct:+.3f}%"

    print(f"[orders] ✅ LIVE buy filled: {filled} {symbol.split('/')[0]} "
          f"@ avg {avg_price} (order id {order.get('id')}) | plan: {plan}{slippage_note}")
    return {
        'id': order.get('id'), 'symbol': symbol,
        'filled_amount': filled, 'avg_price': avg_price, 'dry_run': False,
        'expected_price': expected_price, 'take_profit': take_profit, 'stop_loss': stop_loss,
    }


def place_market_sell(symbol: str, base_amount: float):
    """
    Sells `base_amount` of the base asset at market (e.g.
    base_amount=0.5 on 'ZEC/USDT' sells 0.5 ZEC).
    """
    if DRY_RUN:
        print(f"[orders] 🧪 DRY_RUN sell: {base_amount} {symbol.split('/')[0]} (no real order placed)")
        return {
            'id': 'dry-run', 'symbol': symbol,
            'filled_amount': None, 'avg_price': None, 'dry_run': True,
        }

    exchange = _get_exchange()
    order = exchange.create_order(
        symbol=symbol, type='market', side='sell', amount=base_amount,
    )
    filled = float(order.get('filled', 0) or 0)
    cost   = float(order.get('cost', 0) or 0)
    avg_price = (cost / filled) if filled else None
    print(f"[orders] ✅ LIVE sell filled: {filled} {symbol.split('/')[0]} "
          f"@ avg {avg_price} (order id {order.get('id')})")
    return {
        'id': order.get('id'), 'symbol': symbol,
        'filled_amount': filled, 'avg_price': avg_price, 'dry_run': False,
    }