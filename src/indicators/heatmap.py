"""
Multi-Exchange Liquidation Heatmap & Real-Time Cluster Parser.
Supported: Binance USDS-M, Bybit Linear (V5), OKX Swap (V5).

Public API (unchanged from the Binance-only version):
    get_tracker(symbol), analyze_liquidations(tracker), get_liq_levels(tracker, price),
    get_liq_data(symbol, current_price)
"""

import json
import logging
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import websocket  # pip install websocket-client
except ImportError:
    websocket = None

log = logging.getLogger("liq")

_MAX_EVENTS = 1500

# Binance's forceOrder stream is snapshot-sampled (max 1 event/sec/symbol), so it
# under-reports vs. Bybit/OKX. Tune these if you want to compensate; 1.0 = raw.
VENUE_WEIGHTS: Dict[str, float] = {"Binance": 1.0, "Bybit": 1.0, "OKX": 1.0}

_STABLE_LIFETIME = 30   # seconds a connection must live to reset backoff
_MAX_BACKOFF = 60
_APP_PING_EVERY = 20    # seconds, for venues that want application-level pings


def _normalize_symbol(symbol: str) -> str:
    """Accepts 'SOL', 'SOL/USDT', 'SOL-USDT' or 'SOLUSDT'; returns uppercase base ('SOL')."""
    s = symbol.upper().replace("/", "").replace("-", "")
    return s[:-4] if s.endswith("USDT") and len(s) > 4 else s


class MultiExchangeLiquidationTracker:
    """
    Rolling window of unified liquidation events across Binance, Bybit and OKX.
    One thread per venue, each auto-reconnecting with exponential backoff.
    """

    def __init__(self, symbol: str = "SOL"):
        if websocket is None:
            raise RuntimeError("websocket-client is not installed: pip install websocket-client")

        self.base_symbol = _normalize_symbol(symbol)
        self.events: deque = deque(maxlen=_MAX_EVENTS)
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._started = False

        self.connections: Dict[str, bool] = {"Binance": False, "Bybit": False, "OKX": False}
        self.last_event_at: Optional[datetime] = None
        self._ws: Dict[str, Any] = {}

        # OKX reports size in contracts; ctVal (coins per contract) is fetched lazily.
        self.okx_inst = f"{self.base_symbol}-USDT-SWAP"
        self.okx_ct_val: Optional[float] = None

        self._threads = [
            threading.Thread(target=self._run_binance, daemon=True, name=f"liq-binance-{self.base_symbol}"),
            threading.Thread(target=self._run_bybit, daemon=True, name=f"liq-bybit-{self.base_symbol}"),
            threading.Thread(target=self._run_okx, daemon=True, name=f"liq-okx-{self.base_symbol}"),
        ]

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if not self._started:
            self._started = True
            for t in self._threads:
                t.start()
        return self

    def stop(self):
        self._stop_event.set()
        for ws in list(self._ws.values()):
            try:
                ws.close()
            except Exception:
                pass

    @property
    def _stopped(self) -> bool:
        return self._stop_event.is_set()

    def is_connected(self) -> bool:
        return any(self.connections.values())

    def _set_status(self, venue: str, status: bool):
        if self.connections.get(venue) != status:
            log.info("[liq:%s] %s %s", self.base_symbol, venue, "connected" if status else "disconnected")
        self.connections[venue] = status

    # ------------------------------------------------------------------ events
    def _push_event(self, venue: str, side: str, price: float, qty: float):
        """side is normalized to 'LONG_LIQ' or 'SHORT_LIQ'."""
        now = datetime.now()
        event = {
            "time": now,
            "exchange": venue,
            "side": side,
            "price": price,
            "qty": qty,
            "notional": price * qty,
        }
        with self.lock:
            self.events.append(event)
            self.last_event_at = now

    def snapshot(self, window_minutes: int = 60) -> List[Dict[str, Any]]:
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        with self.lock:
            return [e for e in self.events if e["time"] >= cutoff]

    # ------------------------------------------------------------------ generic socket runner
    def _run_ws(
        self,
        venue: str,
        url: str,
        on_message: Callable[[Any, str], None],
        sub_payload: Optional[str] = None,
        app_ping: Optional[str] = None,
        before_connect: Optional[Callable[[], bool]] = None,
    ):
        """
        Connect -> (subscribe) -> run until dropped -> backoff -> repeat.
        Backoff only resets if the previous connection stayed up for a while,
        so a venue that accepts then immediately drops us still backs off.
        """
        backoff = 1

        def on_open(ws):
            self._set_status(venue, True)
            if sub_payload:
                ws.send(sub_payload)
            if app_ping:
                threading.Thread(target=self._keepalive, args=(ws, app_ping), daemon=True).start()

        def on_close(ws, code, msg):
            self._set_status(venue, False)

        def on_error(ws, err):
            log.warning("[liq:%s] %s socket error: %s", self.base_symbol, venue, err)

        while not self._stopped:
            lived = 0.0
            try:
                if before_connect and not before_connect():
                    raise RuntimeError("pre-connect step failed")
                ws = websocket.WebSocketApp(
                    url,
                    on_open=on_open,
                    on_close=on_close,
                    on_error=on_error,
                    on_message=on_message,
                )
                self._ws[venue] = ws
                started = time.time()
                ws.run_forever(ping_interval=20, ping_timeout=10)
                lived = time.time() - started
            except Exception as e:
                log.warning("[liq:%s] %s runner error: %s", self.base_symbol, venue, e)

            self._set_status(venue, False)
            if self._stopped:
                break
            self._stop_event.wait(backoff)
            backoff = 1 if lived > _STABLE_LIFETIME else min(backoff * 2, _MAX_BACKOFF)

    def _keepalive(self, ws, payload: str):
        while not self._stopped and getattr(ws, "keep_running", False):
            if self._stop_event.wait(_APP_PING_EVERY):
                return
            try:
                ws.send(payload)
            except Exception:
                return

    # ------------------------------------------------------------------ Binance
    def _run_binance(self):
        url = f"wss://fstream.binance.com/ws/{self.base_symbol.lower()}usdt@forceOrder"
        self._run_ws("Binance", url, self._handle_binance_msg)

    def _handle_binance_msg(self, ws, message):
        try:
            o = json.loads(message)["o"]
            # SELL order = a long was liquidated; BUY = a short was liquidated
            side = "LONG_LIQ" if o["S"] == "SELL" else "SHORT_LIQ"
            self._push_event("Binance", side, float(o["ap"]), float(o["z"]))
        except Exception as e:
            log.debug("[liq] binance parse error: %s", e)

    # ------------------------------------------------------------------ Bybit
    def _run_bybit(self):
        sub = json.dumps({"op": "subscribe", "args": [f"allLiquidation.{self.base_symbol}USDT"]})
        self._run_ws(
            "Bybit",
            "wss://stream.bybit.com/v5/public/linear",
            self._handle_bybit_msg,
            sub_payload=sub,
            app_ping=json.dumps({"op": "ping"}),
        )

    def _handle_bybit_msg(self, ws, message):
        try:
            msg = json.loads(message)
            if msg.get("success") is False:
                log.warning("[liq] bybit subscribe failed: %s", msg.get("ret_msg"))
                return
            data = msg.get("data")
            if not data:
                return
            items = data if isinstance(data, list) else [data]
            for item in items:
                # Bybit: S == "Buy" means a long position was liquidated
                side = "LONG_LIQ" if item["S"] == "Buy" else "SHORT_LIQ"
                self._push_event("Bybit", side, float(item["p"]), float(item["v"]))
        except Exception as e:
            log.debug("[liq] bybit parse error: %s", e)

    # ------------------------------------------------------------------ OKX
    def _fetch_okx_ct_val(self) -> bool:
        if self.okx_ct_val:
            return True
        try:
            url = f"https://www.okx.com/api/v5/public/instruments?instType=SWAP&instId={self.okx_inst}"
            with urllib.request.urlopen(url, timeout=5) as r:
                self.okx_ct_val = float(json.load(r)["data"][0]["ctVal"])
            log.info("[liq:%s] OKX ctVal = %s", self.base_symbol, self.okx_ct_val)
            return True
        except Exception as e:
            log.warning("[liq:%s] could not fetch OKX ctVal (%s); is %s listed?",
                        self.base_symbol, e, self.okx_inst)
            return False

    def _run_okx(self):
        sub = json.dumps({"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]})
        self._run_ws(
            "OKX",
            "wss://ws.okx.com:8443/ws/v5/public",
            self._handle_okx_msg,
            sub_payload=sub,
            app_ping="ping",
            before_connect=self._fetch_okx_ct_val,
        )

    def _handle_okx_msg(self, ws, message):
        if message == "pong":
            return
        try:
            msg = json.loads(message)
            if msg.get("event") == "error":
                log.warning("[liq] okx error: %s", msg.get("msg"))
                return
            if not self.okx_ct_val:
                return
            for item in msg.get("data", []):
                if item.get("instId") != self.okx_inst:
                    continue
                for d in item.get("details", []):
                    # `side` works for both hedge and net mode (posSide may be "net").
                    # sell = long liquidated, buy = short liquidated
                    side = "LONG_LIQ" if d["side"].lower() == "sell" else "SHORT_LIQ"
                    self._push_event("OKX", side, float(d["bkPx"]), float(d["sz"]) * self.okx_ct_val)
        except Exception as e:
            log.debug("[liq] okx parse error: %s", e)


# ====================================================================== analysis
def analyze_liquidations(
    tracker: MultiExchangeLiquidationTracker, window_minutes: int = 60
) -> Optional[Dict[str, Any]]:
    recent = tracker.snapshot(window_minutes)
    if not recent:
        return None

    def w(e):  # venue-weighted notional
        return e["notional"] * VENUE_WEIGHTS.get(e["exchange"], 1.0)

    long_liq = sum(w(e) for e in recent if e["side"] == "LONG_LIQ")
    short_liq = sum(w(e) for e in recent if e["side"] == "SHORT_LIQ")
    total_liq = long_liq + short_liq
    liq_ratio = long_liq / (total_liq + 1e-9)

    if liq_ratio > 0.65:
        bias, comment = "BEARISH", "Heavy multi-exchange long liquidations: downward pressure"
    elif liq_ratio < 0.35:
        bias, comment = "BULLISH", "Heavy multi-exchange short liquidations: upward squeeze"
    else:
        bias, comment = "NEUTRAL", "Balanced liquidations across venues"

    by_venue: Dict[str, float] = {}
    for e in recent:
        by_venue[e["exchange"]] = by_venue.get(e["exchange"], 0.0) + e["notional"]

    events_per_min = len(recent) / max(window_minutes, 1)
    return {
        "bias": bias,
        "comment": comment,
        "long_liq_usd": long_liq,
        "short_liq_usd": short_liq,
        "liq_ratio": liq_ratio,
        "liq_spike": events_per_min > 3,
        "total_liq_usd": total_liq,
        "event_count": len(recent),
        "by_venue_usd": by_venue,
        "venue_status": dict(tracker.connections),
    }


def get_liq_levels(
    tracker: MultiExchangeLiquidationTracker,
    current_price: Optional[float] = None,
    window_minutes: int = 180,
    cluster_pct: float = 0.3,
) -> List[Dict[str, Any]]:
    """Approximates liquidation zones by clustering recent liquidation prices (all venues)."""
    if not current_price:
        return []

    events = tracker.snapshot(window_minutes)
    if not events:
        return []

    clusters: List[Dict[str, Any]] = []
    for e in sorted(events, key=lambda x: x["price"]):
        amount = e["notional"] * VENUE_WEIGHTS.get(e["exchange"], 1.0)
        placed = False
        for c in clusters:
            if abs(e["price"] - c["price"]) / c["price"] * 100 <= cluster_pct:
                c["amount"] += amount
                c["price"] = (c["price"] + e["price"]) / 2
                placed = True
                break
        if not placed:
            clusters.append({"price": e["price"], "amount": amount})

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


# ====================================================================== registry
_trackers: Dict[str, MultiExchangeLiquidationTracker] = {}
_tracker_lock = threading.Lock()


def get_tracker(symbol: str) -> MultiExchangeLiquidationTracker:
    key = _normalize_symbol(symbol)
    with _tracker_lock:
        if key not in _trackers:
            _trackers[key] = MultiExchangeLiquidationTracker(symbol=key).start()
        return _trackers[key]


def get_liq_data(
    symbol: str = "SOL", current_price: Optional[float] = None
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (liquidation analysis, nearby liquidation clusters)."""
    tracker = get_tracker(symbol)

    waited = 0.0
    while not tracker.is_connected() and waited < 5:
        time.sleep(0.5)
        waited += 0.5

    return analyze_liquidations(tracker), get_liq_levels(tracker, current_price)