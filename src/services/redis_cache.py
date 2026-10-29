"""
Shared state storage wrapper supporting Redis caching with durable SQLite persistence.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any, Iterator, Tuple

from config.settings import settings

try:
    import redis
except ImportError:
    redis = None

_MAX_REDIS_RETRY_DELAY = 30.0
_REDIS_RECONNECT_EVERY = 60.0


class StateStorage:
    """
    Unified state storage combining Redis fast memory cache with SQLite persistent storage.
    SQLite is the source of truth for bot state; Redis is a cache and the shared key-value bus.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        redis_url: Optional[str] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        connect_retries: Optional[int] = None,
    ):
        self.db_path = db_path or settings.DB_PATH
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_host = redis_host or settings.REDIS_HOST
        self.redis_port = redis_port or settings.REDIS_PORT
        self.redis_client = None
        self._last_redis_attempt = 0.0
        self._init_redis(connect_retries)
        self.init_db()

    def _make_redis_client(self) -> Any:
        if self.redis_url:
            return redis.from_url(self.redis_url, decode_responses=True, socket_timeout=2)
        return redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True, socket_timeout=2)

    def _connect_redis_once(self) -> bool:
        self._last_redis_attempt = time.monotonic()
        try:
            client = self._make_redis_client()
            client.ping()
            self.redis_client = client
            return True
        except Exception:
            self.redis_client = None
            return False

    def _init_redis(self, connect_retries: Optional[int] = None):
        """
        Connects to Redis with retries. Retries only apply when REDIS_URL is configured
        (Docker), so local runs without Redis don't stall at startup.
        """
        if redis is None:
            return

        if connect_retries is None:
            connect_retries = settings.REDIS_CONNECT_RETRIES if self.redis_url else 1
        attempts = max(1, connect_retries)
        for attempt in range(1, attempts + 1):
            if self._connect_redis_once():
                print("[StateStorage] ⚡ Connected to Redis shared cache")
                return
            if attempt < attempts:
                time.sleep(min(settings.REDIS_RETRY_DELAY_SEC * attempt, _MAX_REDIS_RETRY_DELAY))
        print(f"[StateStorage] ⚠️ Redis unavailable after {attempts} tries — using SQLite only")

    def _ensure_redis(self):
        """Lazy reconnect, at most once per _REDIS_RECONNECT_EVERY seconds."""
        if self.redis_client is not None or redis is None:
            return
        if time.monotonic() - self._last_redis_attempt < _REDIS_RECONNECT_EVERY:
            return
        if self._connect_redis_once():
            print("[StateStorage] ⚡ Reconnected to Redis shared cache")

    def _redis_failed(self, what: str, e: Exception):
        print(f"[StateStorage] Redis {what} error: {e}")
        if redis is not None and isinstance(e, (redis.ConnectionError, redis.TimeoutError)):
            self.redis_client = None   # lazy reconnect takes over

    def redis_connected(self) -> bool:
        return self.redis_client is not None

    @contextmanager
    def get_db_connection(self) -> Iterator[sqlite3.Connection]:
        """Yields a connection that commits on success and is always closed
        (sqlite3's own `with conn:` only commits, it never closes)."""
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            with conn:
                yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initializes SQLite schema for state management and trade history."""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS symbol_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    open_positions TEXT NOT NULL,
                    daily_trades INTEGER NOT NULL,
                    daily_pnl REAL NOT NULL,
                    day INTEGER NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    close_reason TEXT NOT NULL,
                    closed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._migrate_trade_date(cursor)
            conn.commit()

    @staticmethod
    def _migrate_trade_date(cursor: sqlite3.Cursor):
        """Adds symbol_state.trade_date (YYYY-MM-DD) to old databases and backfills it from updated_at."""
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(symbol_state)")}
        if "trade_date" not in columns:
            try:
                cursor.execute("ALTER TABLE symbol_state ADD COLUMN trade_date TEXT")
            except sqlite3.OperationalError as e:
                # Several bots share this DB; another one may have migrated it first
                if "duplicate column" not in str(e).lower():
                    raise
        cursor.execute("""
            UPDATE symbol_state SET trade_date = substr(updated_at, 1, 10)
            WHERE trade_date IS NULL AND updated_at IS NOT NULL
        """)

    # ── State Persistence (symbol_state) ────────────────────────

    def save_bot_state(
        self,
        symbol: str,
        open_positions: List[Dict[str, Any]],
        daily_trades: int,
        daily_pnl: float,
        trade_date: str,
    ):
        """Saves runtime state to SQLite (source of truth) and updates the Redis cache."""
        now_iso = datetime.now().isoformat()
        positions_json = json.dumps(open_positions)
        day = int(trade_date[8:10])  # legacy column, kept for old readers

        # 1. Update SQLite
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO symbol_state (symbol, open_positions, daily_trades, daily_pnl, day, trade_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    open_positions=excluded.open_positions,
                    daily_trades=excluded.daily_trades,
                    daily_pnl=excluded.daily_pnl,
                    day=excluded.day,
                    trade_date=excluded.trade_date,
                    updated_at=excluded.updated_at
                """,
                (symbol, positions_json, daily_trades, daily_pnl, day, trade_date, now_iso),
            )
            conn.commit()

        # 2. Update Redis cache; a failure here never fails the save
        self._ensure_redis()
        if self.redis_client:
            try:
                state_data = {
                    "open_positions": positions_json,
                    "daily_trades": daily_trades,
                    "daily_pnl": daily_pnl,
                    "day": day,
                    "trade_date": trade_date,
                    "updated_at": now_iso,
                }
                self.redis_client.set(f"bot_state:{symbol}", json.dumps(state_data))
            except Exception as e:
                self._redis_failed("cache write", e)

    @staticmethod
    def _parse_ts(value: Any) -> datetime:
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return datetime.min

    def load_bot_state(self, symbol: str) -> Tuple[List[Dict[str, Any]], int, float, Optional[str], str]:
        """
        Returns (open_positions, daily_trades, daily_pnl, trade_date, source).
        Reads SQLite always and the Redis copy if present; the newer updated_at wins.
        """
        candidates: Dict[str, Dict[str, Any]] = {}

        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT open_positions, daily_trades, daily_pnl, trade_date, updated_at FROM symbol_state WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()
            if row:
                candidates["sqlite"] = {
                    "open_positions": json.loads(row["open_positions"]),
                    "daily_trades": int(row["daily_trades"]),
                    "daily_pnl": float(row["daily_pnl"]),
                    "trade_date": row["trade_date"] or (str(row["updated_at"])[:10] if row["updated_at"] else None),
                    "updated_at": row["updated_at"],
                }

        self._ensure_redis()
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"bot_state:{symbol}")
                if cached:
                    data = json.loads(cached)
                    updated_at = data.get("updated_at")
                    candidates["redis"] = {
                        "open_positions": json.loads(data["open_positions"]),
                        "daily_trades": int(data["daily_trades"]),
                        "daily_pnl": float(data["daily_pnl"]),
                        "trade_date": data.get("trade_date") or (str(updated_at)[:10] if updated_at else None),
                        "updated_at": updated_at,
                    }
            except Exception as e:
                self._redis_failed("cache read", e)

        if not candidates:
            return [], 0, 0.0, None, "none"

        # Ties go to SQLite, the source of truth
        source = max(candidates, key=lambda k: (self._parse_ts(candidates[k]["updated_at"]), k == "sqlite"))
        chosen = candidates[source]

        if len(candidates) == 2:
            def summary(c):
                return (c["trade_date"], c["daily_trades"], round(c["daily_pnl"], 6), len(c["open_positions"]))
            if summary(candidates["sqlite"]) != summary(candidates["redis"]):
                print(
                    f"[StateStorage] ⚠️ {symbol}: Redis state {summary(candidates['redis'])} @ {candidates['redis']['updated_at']} "
                    f"differs from SQLite {summary(candidates['sqlite'])} @ {candidates['sqlite']['updated_at']} — using {source}"
                )

        return chosen["open_positions"], chosen["daily_trades"], chosen["daily_pnl"], chosen["trade_date"], source

    # ── Trade History (trade_history) ───────────────────────────

    def record_closed_trade(
        self,
        symbol: str,
        action: str,
        entry: float,
        exit_price: float,
        pnl_pct: float,
        reason: str,
    ):
        """Permanently records closed trade into history."""
        now_iso = datetime.now().isoformat()
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trade_history (symbol, action, entry_price, exit_price, pnl_pct, close_reason, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, action, entry, exit_price, pnl_pct, reason, now_iso),
            )
            conn.commit()

    def get_daily_summary(self) -> Tuple[List[Dict[str, Any]], float, int]:
        """Queries all closed trades for today across all symbols."""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    symbol,
                    SUM(pnl_pct) as total_pnl,
                    COUNT(id) as total_trades,
                    SUM(CASE WHEN pnl_pct >= 0 THEN 1 ELSE 0 END) as win_trades
                FROM trade_history
                WHERE DATE(closed_at) = DATE('now', 'localtime')
                GROUP BY symbol
            """)
            rows = cursor.fetchall()

        summary = []
        overall_pnl = 0.0
        overall_trades = 0

        for row in rows:
            pnl = row["total_pnl"] or 0.0
            trades = row["total_trades"]
            summary.append({
                "symbol": row["symbol"],
                "pnl_pct": pnl,
                "trades": trades,
                "wins": row["win_trades"],
            })
            overall_pnl += pnl
            overall_trades += trades

        return summary, overall_pnl, overall_trades

    def get_last_close_time(self, symbol: str, reason_prefix: str) -> Optional[datetime]:
        """Returns timestamp of most recent trade closed matching reason_prefix."""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT closed_at FROM trade_history
                WHERE symbol = ? AND close_reason LIKE ?
                ORDER BY id DESC LIMIT 1
            """, (symbol, f"{reason_prefix}%"))
            row = cursor.fetchone()
            if row and row["closed_at"]:
                try:
                    return datetime.fromisoformat(row["closed_at"])
                except Exception:
                    return None
        return None

    def get_last_sl_time(self, symbol: str) -> Optional[datetime]:
        return self.get_last_close_time(symbol, "SL Hit")

    def get_last_tp_time(self, symbol: str) -> Optional[datetime]:
        # The trader applies the TP cooldown after profit-lock exits too
        times = [t for t in (self.get_last_close_time(symbol, "TP Hit"),
                             self.get_last_close_time(symbol, "Profit Lock Hit")) if t]
        return max(times) if times else None

    # ── Arbitrary Key-Value Cache ───────────────────────────────

    def set_shared_value(self, key: str, value: Any, expire_seconds: Optional[int] = None):
        """Sets a key in Redis or local in-memory fallback."""
        self._ensure_redis()
        if self.redis_client:
            try:
                val_str = json.dumps(value) if not isinstance(value, str) else value
                self.redis_client.set(key, val_str, ex=expire_seconds)
            except Exception as e:
                self._redis_failed(f"set_shared_value({key})", e)

    def get_shared_value(self, key: str) -> Optional[Any]:
        """Gets a key from Redis."""
        self._ensure_redis()
        if self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    try:
                        return json.loads(val)
                    except Exception:
                        return val
            except Exception as e:
                self._redis_failed(f"get_shared_value({key})", e)
        return None


# ── Global Storage Instance & Functional API ─────────────────────
_default_storage: Optional[StateStorage] = None


def get_storage() -> StateStorage:
    global _default_storage
    if _default_storage is None:
        _default_storage = StateStorage()
    return _default_storage


def init_db():
    get_storage().init_db()

def save_bot_state(symbol: str, open_positions: list, daily_trades: int, daily_pnl: float, trade_date: str):
    get_storage().save_bot_state(symbol, open_positions, daily_trades, daily_pnl, trade_date)

def load_bot_state(symbol: str):
    return get_storage().load_bot_state(symbol)

def redis_connected() -> bool:
    return get_storage().redis_connected()

def record_closed_trade(symbol: str, action: str, entry: float, exit_price: float, pnl_pct: float, reason: str):
    get_storage().record_closed_trade(symbol, action, entry, exit_price, pnl_pct, reason)

def get_daily_summary():
    return get_storage().get_daily_summary()

def get_last_sl_time(symbol: str):
    return get_storage().get_last_sl_time(symbol)

def get_last_tp_time(symbol: str):
    return get_storage().get_last_tp_time(symbol)

def set_shared_value(key: str, value: Any, expire_seconds: Optional[int] = None):
    get_storage().set_shared_value(key, value, expire_seconds)

def get_shared_value(key: str):
    return get_storage().get_shared_value(key)

