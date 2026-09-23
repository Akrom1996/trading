"""
Free replacement for the Coinglass-based liquidation module.

Data source: Binance USDS-M Futures public `forceOrder` WebSocket stream.
No API key needed -- this is public market data.

Note: Binance deprecated the public REST endpoint for market-wide
liquidation history (GET /fapi/v1/allForceOrders no longer accepts
requests), so there is no second live REST source to fall back to.
Resilience here comes from auto-reconnect + an in-memory rolling
window, not from a second data provider.

Install dependency:
    pip install websocket-client
"""

import json
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import websocket  # pip install websocket-client

BINANCE_WS_BASE = "wss://fstream.binance.com/ws"
_MAX_EVENTS = 500  # rolling window of recent liquidation events per symbol


def _normalize_symbol(symbol: str) -> str:
    """Accepts 'ONE' or 'ONEUSDT' and returns the Binance futures symbol."""
    symbol = symbol.upper()
    return symbol if symbol.endswith("USDT") else f"{symbol}USDT"


class LiquidationTracker:
    """
    Maintains a rolling window of real liquidation events for one symbol,
    fed by Binance's public forceOrder websocket stream. Runs in a
    background thread and auto-reconnects with exponential backoff.
    """

    def __init__(self, symbol="ONEUSDT"):
        self.symbol = _normalize_symbol(symbol)
        self.events = deque(maxlen=_MAX_EVENTS)
        self.lock = threading.Lock()
        self.ws = None
        self.connected = False
        self.last_event_at = None
        self._stop = False
        self._thread = threading.Thread(target=self._run_forever, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
        if self.ws:
            self.ws.close()

    # ── internal websocket plumbing ──────────────────────
    def _run_forever(self):
        backoff = 1
        while not self._stop:
            try:
                url = f"{BINANCE_WS_BASE}/{self.symbol.lower()}@forceOrder"
                self.ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_close=self._on_close,
                    on_error=self._on_error,
                )
                backoff = 1  # reset backoff after a run_forever attempt starts
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"[liq:{self.symbol}] websocket crashed: {e}")

            if self._stop:
                break
            print(f"[liq:{self.symbol}] reconnecting in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def _on_open(self, ws):
        self.connected = True
        print(f"[liq:{self.symbol}] connected")

    def _on_close(self, ws, code, msg):
        self.connected = False
        print(f"[liq:{self.symbol}] disconnected ({code}: {msg})")

    def _on_error(self, ws, error):
        print(f"[liq:{self.symbol}] error: {error}")

    def _on_message(self, ws, message):
        try:
            o = json.loads(message)["o"]
            event = {
                "time": datetime.fromtimestamp(o["T"] / 1000),
                # SELL forceOrder = a long position got liquidated
                # BUY  forceOrder = a short position got liquidated
                "side": o["S"],
                "price": float(o["ap"]),
                "qty": float(o["z"]),
                "notional": float(o["ap"]) * float(o["z"]),
            }
            with self.lock:
                self.events.append(event)
                self.last_event_at = event["time"]
        except Exception as e:
            print(f"[liq:{self.symbol}] parse error: {e}")

    # ── public read access ────────────────────────────────
    def snapshot(self, window_minutes=60):
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        with self.lock:
            return [e for e in self.events if e["time"] >= cutoff]


# ── analysis (mirrors the old Coinglass-based logic) ──────
def analyze_liquidations(tracker: LiquidationTracker, window_minutes=60):
    recent = tracker.snapshot(window_minutes)
    if not recent:
        return None

    long_liq = sum(e["notional"] for e in recent if e["side"] == "SELL")
    short_liq = sum(e["notional"] for e in recent if e["side"] == "BUY")
    total_liq = long_liq + short_liq
    liq_ratio = long_liq / (total_liq + 1e-9)

    if liq_ratio > 0.65:
        bias, comment = "BEARISH", "Heavy long liquidations — downward pressure"
    elif liq_ratio < 0.35:
        bias, comment = "BULLISH", "Heavy short liquidations — upward pressure"
    else:
        bias, comment = "NEUTRAL", "Balanced liquidations — no clear bias"

    events_per_min = len(recent) / max(window_minutes, 1)
    liq_spike = events_per_min > 2  # tune threshold to taste per symbol

    return {
        "bias": bias,
        "comment": comment,
        "long_liq_usd": long_liq,
        "short_liq_usd": short_liq,
        "liq_ratio": liq_ratio,
        "liq_spike": liq_spike,
        "total_liq_usd": total_liq,
        "event_count": len(recent),
    }


def get_liq_levels(tracker: LiquidationTracker, current_price=None,
                    window_minutes=180, cluster_pct=0.3):
    """
    Approximates liquidation "zones" by clustering actual recent
    liquidation prices. This is NOT a predictive heatmap like Coinglass's
    (which models future liquidation risk from inferred open positions) --
    it only reflects where liquidations have already happened recently.
    """
    if not current_price:
        return []

    events = tracker.snapshot(window_minutes)
    if not events:
        return []

    clusters = []
    for e in sorted(events, key=lambda x: x["price"]):
        placed = False
        for c in clusters:
            if abs(e["price"] - c["price"]) / c["price"] * 100 <= cluster_pct:
                c["amount"] += e["notional"]
                c["price"] = (c["price"] + e["price"]) / 2
                placed = True
                break
        if not placed:
            clusters.append({"price": e["price"], "amount": e["notional"]})

    nearby = []
    for c in clusters:
        dist = abs(c["price"] - current_price) / current_price * 100
        if dist <= 5:
            nearby.append({
                "price": round(c["price"], 4),
                "amount": round(c["amount"], 2),
                "distance": round(dist, 2),
                "direction": "ABOVE" if c["price"] > current_price else "BELOW",
            })
    return sorted(nearby, key=lambda x: x["distance"])[:5]


# ── main entry point — call this from start.py ─────────────
_trackers = {}
_tracker_lock = threading.Lock()


def _get_tracker(symbol: str) -> LiquidationTracker:
    key = _normalize_symbol(symbol)
    with _tracker_lock:
        if key not in _trackers:
            _trackers[key] = LiquidationTracker(symbol=key).start()
        return _trackers[key]


def get_liq_data(symbol='ONE', current_price=None):
    """
    Drop-in replacement for the old Coinglass-based get_liq_data().
    Returns (liq_analysis_dict_or_None, levels_list).

    Backed by a live Binance forceOrder websocket instead of a paid API.
    The tracker is created once per symbol and kept running in the
    background across calls, so repeated calls are cheap.
    """
    tracker = _get_tracker(symbol)

    # give the socket a moment to connect on the very first call
    waited = 0.0
    while not tracker.connected and waited < 5:
        time.sleep(0.5)
        waited += 0.5

    liq = analyze_liquidations(tracker)
    levels = get_liq_levels(tracker, current_price)
    return liq, levels