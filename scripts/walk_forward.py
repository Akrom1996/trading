"""
Walk-forward validation of the live entry rule.

Rolling windows over VALIDATION_CANDLES of 5m data: train both models on TRAIN_CANDLES rows,
purge LABEL_HORIZON_CANDLES, test on the next WF_TEST_CANDLES, step WF_STEP_CANDLES. Entries use
the same features, labels, models and EV rule as live, replayed as non-overlapping trades with the
bot's limit-fill rules, and are compared with a baseline that buys every flat candle at the same tier.

    python -m scripts.walk_forward DASH/USDT
"""

import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.exchanges.binance_client import BinanceClient, fetch_ohlcv_history
from src.indicators.technicals import (
    FEATURES,
    LABEL_HORIZON_CANDLES,
    add_features,
    add_labels,
    directional_buy_mask,
    ev_entry_rule,
    median_tier,
    simulate_trades,
    summarize_trades,
    train_barrier_model,
    train_model,
)

WF_N_JOBS = 2  # leave CPU for the trading loop when this runs in a background thread


def _ev_entries(model: Any, scaler: Any, encoder: Any, X: pd.DataFrame):
    probs = model.predict_proba(scaler.transform(X))
    enter = np.zeros(len(X), dtype=bool)
    tiers = np.ones(len(X), dtype=int)
    for row, row_probs in enumerate(probs):
        rule = ev_entry_rule({int(c): float(p) for c, p in zip(encoder.classes_, row_probs)})
        enter[row] = rule['enter']
        tiers[row] = rule['chosen_tier']
    return enter, tiers


def _replay(feats: pd.DataFrame, windows: List[Dict[str, Any]], mask: np.ndarray,
            tiers: np.ndarray) -> List[List[Dict[str, Any]]]:
    """Non-overlapping across windows too: a window starts after the previous trade has exited."""
    per_window, next_free = [], 0
    for w in windows:
        trades = simulate_trades(feats, mask, tiers, max(w['test_start'], next_free), w['test_end'])
        for t in trades:
            next_free = max(next_free, t.get('exit_idx', t['signal_idx']) + 1)
        per_window.append(trades)
    return per_window


def run_walk_forward(symbol: str, df: Optional[pd.DataFrame] = None,
                     client: Optional[BinanceClient] = None, verbose: bool = True) -> Dict[str, Any]:
    if df is None:
        client = client or BinanceClient()   # own ccxt instance: safe to call from a background thread
        df = fetch_ohlcv_history(symbol, '5m', settings.VALIDATION_CANDLES, client=client)
    feats = df if all(c in df.columns for c in FEATURES) else add_features(df)
    feats = feats.reset_index(drop=True)
    n = len(feats)

    train_len, test_len, step = settings.TRAIN_CANDLES, settings.WF_TEST_CANDLES, settings.WF_STEP_CANDLES
    mask = np.zeros(n, dtype=bool)
    tiers = np.ones(n, dtype=int)
    windows: List[Dict[str, Any]] = []

    start = 0
    while start + train_len + LABEL_HORIZON_CANDLES + test_len <= n:
        train = feats.iloc[start:start + train_len].reset_index(drop=True)
        test_start = start + train_len + LABEL_HORIZON_CANDLES   # purge: no train label sees test candles
        test_end = test_start + test_len

        model, scaler, encoder, meta = train_model(add_labels(train), oos=False, verbose=False, n_jobs=WF_N_JOBS)
        b_model, b_scaler, b_encoder, _ = train_barrier_model(train, oos=False, verbose=False, n_jobs=WF_N_JOBS)

        X = feats.iloc[test_start:test_end][FEATURES]
        buy = directional_buy_mask(model, scaler, encoder, X, meta['base_rate_bullish'])
        enter, chosen = _ev_entries(b_model, b_scaler, b_encoder, X)
        mask[test_start:test_end] = buy & enter
        tiers[test_start:test_end] = chosen
        windows.append({'test_start': test_start, 'test_end': test_end,
                        'from': str(feats['timestamp'].iloc[test_start]) if 'timestamp' in feats else test_start,
                        'to': str(feats['timestamp'].iloc[test_end - 1]) if 'timestamp' in feats else test_end - 1})
        if verbose:
            print(f"  [{symbol}] window {len(windows)}: {windows[-1]['from']} → {windows[-1]['to']} "
                  f"buy={int(buy.sum())} ev={int(enter.sum())} both={int((buy & enter).sum())}")
        start += step

    model_trades = _replay(feats, windows, mask, tiers)
    all_model = [t for ts in model_trades for t in ts]
    base_tier = median_tier(all_model)
    baseline_trades = _replay(feats, windows, np.ones(n, dtype=bool), np.full(n, base_tier))
    all_base = [t for ts in baseline_trades for t in ts]

    stats = summarize_trades(all_model)
    base = summarize_trades(all_base)
    edge = (stats['avg_net'] - base['avg_net']
            if stats['avg_net'] is not None and base['avg_net'] is not None else None)

    per_window = []
    for w, ts in zip(windows, model_trades):
        ws = summarize_trades(ts)
        per_window.append({'from': w['from'], 'to': w['to'], 'trades': ws['trades'],
                           'avg_net': ws['avg_net'], 'total_net': ws['total_net']})
    with_trades = [w for w in per_window if w['trades'] > 0]

    stats.update({
        'symbol': symbol,
        'run_at': datetime.now().isoformat(timespec='seconds'),
        'candles': n,
        'windows': len(windows),
        'windows_with_trades': len(with_trades),
        'profitable_window_share': (round(sum(w['total_net'] > 0 for w in with_trades) / len(with_trades), 4)
                                    if with_trades else None),
        'baseline_tier': base_tier,
        'baseline_trades': base['trades'],
        'baseline_win_rate': base['win_rate'],
        'baseline_avg_net': base['avg_net'],
        'edge': round(edge, 5) if edge is not None else None,
        'per_window': per_window,
    })
    return stats


def format_summary(stats: Dict[str, Any]) -> str:
    def pct(v, signed=True):
        return "n/a" if v is None else (f"{v*100:+.2f}%" if signed else f"{v*100:.0f}%")
    lines = [
        f"Walk-forward {stats['symbol']}  ({stats['candles']} candles, {stats['windows']} windows, run {stats['run_at']})",
        f"  {'':10s} {'trades':>7s} {'win':>6s} {'avg net':>9s} {'total':>9s}",
        f"  {'model':10s} {stats['trades']:>7d} {pct(stats['win_rate'], False):>6s} "
        f"{pct(stats['avg_net']):>9s} {pct(stats['total_net']):>9s}",
        f"  {'baseline':10s} {stats['baseline_trades']:>7d} {pct(stats['baseline_win_rate'], False):>6s} "
        f"{pct(stats['baseline_avg_net']):>9s}   (tier {stats['baseline_tier']}, every flat candle)",
        f"  edge {pct(stats['edge'])} | fill rate {pct(stats['fill_rate'], False)} | "
        f"max drawdown {pct(stats['max_drawdown'], False)} | "
        f"profitable windows {pct(stats['profitable_window_share'], False)} of {stats['windows_with_trades']}",
    ]
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    symbol = argv[1] if len(argv) > 1 else settings.SYMBOL
    print(f"Running walk-forward for {symbol} ({settings.VALIDATION_CANDLES} candles)...")
    stats = run_walk_forward(symbol)
    print(format_summary(stats))
    from src.roles.coin_trader import live_entries_allowed   # lazy: the engine imports this module
    allowed, reason = live_entries_allowed(stats)
    print(f"  Live guard: {'ON' if allowed else 'OFF'} — {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
