"""
Shared state storage wrapper supporting Redis caching with durable SQLite persistence.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from config.settings import settings

try:
    import redis
except ImportError:
    redis = None


class StateStorage:
    """
    Unified state storage combining Redis fast memory cache with SQLite persistent storage.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        redis_url: Optional[str] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
    ):
        self.db_path = db_path or settings.DB_PATH
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_host = redis_host or settings.REDIS_HOST
        self.redis_port = redis_port or settings.REDIS_PORT
        self.redis_client = None
        self._init_redis()
        self.init_db()

    def _init_redis(self):
        """Attempts connection to Redis if client library and config are available."""
        if redis is None:
            return

        try:
            if self.redis_url:
                client = redis.from_url(self.redis_url, decode_responses=True, socket_timeout=2)
            else:
                client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True, socket_timeout=2)
            client.ping()
            self.redis_client = client
            print("[StateStorage] ⚡ Connected to Redis shared cache")
        except Exception as e:
            self.redis_client = None
            # Silently fallback to SQLite without crashing
            # print(f"[StateStorage] Redis unavailable ({e}) -- using SQLite directly")

    def get_db_connection(self) -> sqlite3.Connection:
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

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
            conn.commit()

    # ── State Persistence (symbol_state) ────────────────────────

    def save_bot_state(
        self,
        symbol: str,
        open_positions: List[Dict[str, Any]],
        daily_trades: int,
        daily_pnl: float,
        day: int,
    ):
        """Saves runtime state to SQLite and updates Redis cache."""
        now_iso = datetime.now().isoformat()
        positions_json = json.dumps(open_positions)

        # 1. Update SQLite
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO symbol_state (symbol, open_positions, daily_trades, daily_pnl, day, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    open_positions=excluded.open_positions,
                    daily_trades=excluded.daily_trades,
                    daily_pnl=excluded.daily_pnl,
                    day=excluded.day,
                    updated_at=excluded.updated_at
                """,
                (symbol, positions_json, daily_trades, daily_pnl, day, now_iso),
            )
            conn.commit()

        # 2. Update Redis if connected
        if self.redis_client:
            try:
                state_data = {
                    "open_positions": positions_json,
                    "daily_trades": daily_trades,
                    "daily_pnl": daily_pnl,
                    "day": day,
                    "updated_at": now_iso,
                }
                self.redis_client.set(f"bot_state:{symbol}", json.dumps(state_data))
            except Exception as e:
                print(f"[StateStorage] Redis cache write error: {e}")

    def load_bot_state(self, symbol: str) -> Tuple[List[Dict[str, Any]], int, float, Optional[int]]:
        """Loads state for a symbol, checking Redis first then SQLite."""
        # 1. Try Redis cache
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"bot_state:{symbol}")
                if cached:
                    data = json.loads(cached)
                    return (
                        json.loads(data["open_positions"]),
                        int(data["daily_trades"]),
                        float(data["daily_pnl"]),
                        data.get("day"),
                    )
            except Exception as e:
                print(f"[StateStorage] Redis cache read error: {e}")

        # 2. Fallback to SQLite
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT open_positions, daily_trades, daily_pnl, day FROM symbol_state WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()
            if row:
                return (
                    json.loads(row["open_positions"]),
                    row["daily_trades"],
                    row["daily_pnl"],
                    row["day"],
                )

        return [], 0, 0.0, None

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
        return self.get_last_close_time(symbol, "TP Hit")

    # ── Arbitrary Key-Value Cache ───────────────────────────────

    def set_shared_value(self, key: str, value: Any, expire_seconds: Optional[int] = None):
        """Sets a key in Redis or local in-memory fallback."""
        if self.redis_client:
            try:
                val_str = json.dumps(value) if not isinstance(value, str) else value
                self.redis_client.set(key, val_str, ex=expire_seconds)
            except Exception as e:
                print(f"[StateStorage] set_shared_value error for {key}: {e}")

    def get_shared_value(self, key: str) -> Optional[Any]:
        """Gets a key from Redis."""
        if self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    try:
                        return json.loads(val)
                    except Exception:
                        return val
            except Exception as e:
                print(f"[StateStorage] get_shared_value error for {key}: {e}")
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

def save_bot_state(symbol: str, open_positions: list, daily_trades: int, daily_pnl: float, day: int):
    get_storage().save_bot_state(symbol, open_positions, daily_trades, daily_pnl, day)

def load_bot_state(symbol: str):
    return get_storage().load_bot_state(symbol)

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

