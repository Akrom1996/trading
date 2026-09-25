import os
import sqlite3
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "data/bot_data.db")


def get_connection():
    # Ensure directory exists
    dir_name = os.path.dirname(DB_PATH)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Initializes schema for state management and historical trade logging."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # State table: maintains current open positions and daily state per symbol
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

        # Migration: if table already existed without 'id', migrate it safely
        cursor.execute("PRAGMA table_info(symbol_state)")
        columns = [row["name"] for row in cursor.fetchall()]
        if columns and "id" not in columns:
            cursor.execute("""
                CREATE TABLE symbol_state_migrated (
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
                INSERT INTO symbol_state_migrated (symbol, open_positions, daily_trades, daily_pnl, day, updated_at)
                SELECT symbol, open_positions, daily_trades, daily_pnl, day, updated_at FROM symbol_state
            """)
            cursor.execute("DROP TABLE symbol_state")
            cursor.execute("ALTER TABLE symbol_state_migrated RENAME TO symbol_state")

        # History table: logs every closed trade permanently across all coins
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


def save_bot_state(
    symbol: str, open_positions: list, daily_trades: int, daily_pnl: float, day: int
):
    """Saves live runtime state to SQLite (replaces JSON state save)."""
    import json

    with get_connection() as conn:
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
            (
                symbol,
                json.dumps(open_positions),
                daily_trades,
                daily_pnl,
                day,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()


def load_bot_state(symbol: str):
    """Loads state for a given symbol from SQLite."""
    import json

    with get_connection() as conn:
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


def record_closed_trade(
    symbol: str, action: str, entry: float, exit_price: float, pnl_pct: float, reason: str
):
    """Logs a completed trade into permanent history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trade_history (symbol, action, entry_price, exit_price, pnl_pct, close_reason, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                symbol,
                action,
                entry,
                exit_price,
                pnl_pct,
                reason,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()


def get_daily_summary():
    """Queries all closed trades for today across ALL symbols."""
    with get_connection() as conn:
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
        summary.append(
            {
                "symbol": row["symbol"],
                "pnl_pct": pnl,
                "trades": trades,
                "wins": row["win_trades"],
            }
        )
        overall_pnl += pnl
        overall_trades += trades

    return summary, overall_pnl, overall_trades
