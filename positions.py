"""
Position persistence -- survives restarts (crashes, redeploys, `docker
compose up` after a git push, etc).

Without this, `open_positions` only ever lived in an in-memory list in
main.py's run_bot() loop. Any restart mid-trade orphaned that position:
the bot would have no idea it was still holding something on the
exchange, and the trailing-stop / TP / SL logic would never fire again
for it.

Each symbol gets its own file so parallel containers (SOL/ZEC/BTC)
never touch each other's state.
"""

import json
import os
from datetime import datetime

STATE_DIR = os.getenv('STATE_DIR', '.')


def _positions_path(symbol: str) -> str:
    safe = symbol.replace('/', '').upper()
    return os.path.join(STATE_DIR, f"positions_{safe}.json")


def _serialize(pos: dict) -> dict:
    """datetime-safe copy for json.dump."""
    out = dict(pos)
    for k, v in out.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    return out


def save_positions(symbol: str, open_positions: list, daily_trades: int,
                    daily_pnl: float, day: int):
    path = _positions_path(symbol)
    payload = {
        'open_positions': [_serialize(p) for p in open_positions],
        'daily_trades': daily_trades,
        'daily_pnl': daily_pnl,
        'day': day,
        'saved_at': datetime.now().isoformat(),
    }
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)  # atomic write -- avoids a half-written file
    # if the process dies mid-save


def load_positions(symbol: str):
    """
    Returns (open_positions, daily_trades, daily_pnl, day) or
    ([], 0, 0.0, None) if nothing is saved yet.
    """
    path = _positions_path(symbol)
    if not os.path.exists(path):
        return [], 0, 0.0, None
    try:
        with open(path) as f:
            data = json.load(f)
        return (
            data.get('open_positions', []),
            data.get('daily_trades', 0),
            data.get('daily_pnl', 0.0),
            data.get('day'),
        )
    except (json.JSONDecodeError, OSError) as e:
        print(f"[positions:{symbol}] failed to load saved state: {e} — starting fresh")
        return [], 0, 0.0, None