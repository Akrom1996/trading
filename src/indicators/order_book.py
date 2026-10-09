"""
Order book depth, delta, and imbalance indicators.
Pure Python implementation with zero mandatory third-party dependencies.
"""

from typing import List, Dict, Any, Optional


def calculate_depth_delta(
    bids: List[List[float]],
    asks: List[List[float]],
    depth_levels: int = 20,
) -> Dict[str, Any]:
    """
    Computes order book depth metrics across top N levels.
    bids and asks are lists of [price, amount].
    """
    if not bids or not asks:
        return {
            'bid_depth_usd': 0.0,
            'ask_depth_usd': 0.0,
            'delta_usd': 0.0,
            'imbalance_ratio': 0.0,
            'spread': 0.0,
            'spread_pct': 0.0,
            'bias': 'NEUTRAL',
        }

    top_bids = bids[:depth_levels]
    top_asks = asks[:depth_levels]

    bid_vol_usd = sum(float(p) * float(q) for p, q in top_bids)
    ask_vol_usd = sum(float(p) * float(q) for p, q in top_asks)

    best_bid = float(top_bids[0][0])
    best_ask = float(top_asks[0][0])
    spread = best_ask - best_bid
    mid_price = (best_ask + best_bid) / 2.0
    spread_pct = (spread / mid_price) * 100.0 if mid_price > 0 else 0.0

    delta = bid_vol_usd - ask_vol_usd
    total_depth = bid_vol_usd + ask_vol_usd + 1e-9
    imbalance = delta / total_depth

    if imbalance > 0.15:
        bias = "BULLISH"
    elif imbalance < -0.15:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        'best_bid': best_bid,
        'best_ask': best_ask,
        'bid_depth_usd': round(bid_vol_usd, 2),
        'ask_depth_usd': round(ask_vol_usd, 2),
        'delta_usd': round(delta, 2),
        'imbalance_ratio': round(imbalance, 4),
        'spread': round(spread, 6),
        'spread_pct': round(spread_pct, 4),
        'bias': bias,
        'depth_levels': len(top_bids),
    }


def detect_walls(
    bids: List[List[float]],
    asks: List[List[float]],
    depth_levels: int = 50,
    multiplier: float = 3.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identifies significant buy/sell walls in the order book.
    A wall is defined as an order level whose volume exceeds `multiplier` times the average level volume.
    """
    walls = {'buy_walls': [], 'sell_walls': []}

    top_bids = bids[:depth_levels]
    if top_bids:
        bid_amounts = [float(p) * float(q) for p, q in top_bids]
        mean_bid = sum(bid_amounts) / len(bid_amounts) if bid_amounts else 0.0
        for p, q in top_bids:
            notional = float(p) * float(q)
            if notional > mean_bid * multiplier and notional > 5000:
                walls['buy_walls'].append({
                    'price': float(p),
                    'amount_usd': round(notional, 2),
                    'ratio_to_avg': round(notional / (mean_bid + 1e-9), 2),
                })

    top_asks = asks[:depth_levels]
    if top_asks:
        ask_amounts = [float(p) * float(q) for p, q in top_asks]
        mean_ask = sum(ask_amounts) / len(ask_amounts) if ask_amounts else 0.0
        for p, q in top_asks:
            notional = float(p) * float(q)
            if notional > mean_ask * multiplier and notional > 5000:
                walls['sell_walls'].append({
                    'price': float(p),
                    'amount_usd': round(notional, 2),
                    'ratio_to_avg': round(notional / (mean_ask + 1e-9), 2),
                })

    return walls

