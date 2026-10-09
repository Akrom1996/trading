"""
Binance exchange client wrapping both public market data and private order execution via ccxt.
"""

import time
from typing import Optional, Dict, Any, List
import ccxt
import pandas as pd

from config.settings import settings


class BinanceClient:
    """
    Unified Binance API wrapper for market data and order lifecycle.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        dry_run: Optional[bool] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.BINANCE_API_KEY
        self.api_secret = api_secret if api_secret is not None else settings.BINANCE_API_SECRET
        self.testnet = testnet if testnet is not None else settings.BINANCE_TESTNET
        self.dry_run = dry_run if dry_run is not None else settings.DRY_RUN
        self._exchange: Optional[ccxt.binance] = None

    def get_exchange(self) -> ccxt.binance:
        """Initializes and caches the ccxt Binance exchange instance."""
        if self._exchange is not None:
            return self._exchange

        if not self.dry_run and (not self.api_key or not self.api_secret):
            raise RuntimeError(
                "BINANCE_API_KEY / BINANCE_API_SECRET not set -- cannot place real orders. "
                "Set DRY_RUN=true if you just want to log intended orders without executing them."
            )

        exchange = ccxt.binance({
            'apiKey': self.api_key or '',
            'secret': self.api_secret or '',
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })

        if self.testnet:
            exchange.set_sandbox_mode(True)
            print("[BinanceClient] ⚠️ Running against Binance SPOT TESTNET (fake funds)")
        elif not self.dry_run:
            print("[BinanceClient] 🔴 Running against Binance LIVE account — real funds")

        try:
            exchange.load_markets()
        except Exception as e:
            if not self.dry_run:
                raise RuntimeError(f"Failed to load Binance markets: {e}")

        self._exchange = exchange
        return self._exchange

    # ── Public Market Data ──────────────────────────────────────

    @staticmethod
    def timeframe_to_ms(timeframe: str) -> int:
        units = {'m': 60, 'h': 3600, 'd': 86400}
        return int(timeframe[:-1]) * units[timeframe[-1]] * 1000

    def fetch_ohlcv(self, symbol: str = 'SOL/USDT', timeframe: str = '5m', limit: int = 1000) -> pd.DataFrame:
        """
        Fetches historical OHLCV candles, automatically batching requests if limit > 1000.
        """
        exchange = self.get_exchange()
        all_ohlcv = []
        since = None
        batches = (limit // 1000) + 1

        for i in range(batches):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not ohlcv:
                    break

                all_ohlcv = ohlcv + all_ohlcv
                since = ohlcv[0][0] - (1000 * self.timeframe_to_ms(timeframe))
                time.sleep(0.5)
            except Exception as e:
                print(f"[BinanceClient] Batch {i} fetch error: {e}")
                break

        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        if df.empty:
            return df

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        return df.tail(limit)

    def fetch_current_price(self, symbol: str = 'SOL/USDT') -> float:
        """Fetches the latest traded price for a symbol."""
        exchange = self.get_exchange()
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker['last'])

    def fetch_ticker(self, symbol: str = 'SOL/USDT') -> Dict[str, Any]:
        """Fetches full 24h ticker info."""
        exchange = self.get_exchange()
        return exchange.fetch_ticker(symbol)

    def fetch_order_book(self, symbol: str = 'SOL/USDT', limit: int = 100) -> Dict[str, Any]:
        """Fetches order book snapshot."""
        exchange = self.get_exchange()
        return exchange.fetch_order_book(symbol, limit=limit)

    # ── Private Account & Orders ────────────────────────────────

    def get_available_balance(self, asset: str) -> float:
        """Returns available free balance for an asset (e.g. 'USDT', 'SOL')."""
        if self.dry_run:
            return 10000.0
        exchange = self.get_exchange()
        balance = exchange.fetch_balance()
        return float(balance.get(asset, {}).get('free', 0.0))

    def place_limit_buy(self, symbol: str, limit_price: float, quote_amount: float) -> Dict[str, Any]:
        """
        Places a non-blocking LIMIT buy order.
        """
        exchange = self.get_exchange()

        if not self.dry_run:
            quote_asset = symbol.split('/')[1]
            balance = self.get_available_balance(quote_asset)
            if balance < quote_amount:
                raise RuntimeError(
                    f"Insufficient balance: {balance:.2f} {quote_asset} available, "
                    f"{quote_amount:.2f} {quote_asset} required."
                )

        formatted_price = float(exchange.price_to_precision(symbol, limit_price)) if not self.dry_run else limit_price
        raw_amount = quote_amount / formatted_price
        formatted_amount = float(exchange.amount_to_precision(symbol, raw_amount)) if not self.dry_run else round(raw_amount, 6)

        if self.dry_run:
            print(f"[BinanceClient] 🧪 DRY_RUN limit buy: {formatted_amount} {symbol.split('/')[0]} @ {formatted_price} (~{quote_amount:.2f} USDT)")
            return {
                'id': 'dry-run-limit-buy',
                'symbol': symbol,
                'limit_price': formatted_price,
                'filled_amount': formatted_amount,
                'avg_price': formatted_price,
                'dry_run': True,
                'status': 'closed',
            }

        order = exchange.create_order(
            symbol=symbol, type='limit', side='buy',
            amount=formatted_amount, price=formatted_price,
        )
        filled = float(order.get('filled', 0) or 0)
        print(f"[BinanceClient] 📝 LIVE limit buy placed: {formatted_amount} {symbol.split('/')[0]} "
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

    def check_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Non-blocking status check for a given order ID.
        """
        if order_id == 'dry-run-limit-buy':
            return {'status': 'closed', 'filled_amount': None, 'avg_price': None}

        try:
            exchange = self.get_exchange()
            order_info = exchange.fetch_order(order_id, symbol=symbol)
            filled = float(order_info.get('filled', 0) or 0)
            avg_price = order_info.get('average')
            avg_price = float(avg_price) if avg_price is not None else None
            return {
                'status': order_info.get('status'),
                'filled_amount': filled,
                'avg_price': avg_price,
            }
        except Exception as e:
            print(f"[BinanceClient] check_order_status error for {order_id}: {e}")
            return {'status': 'unknown', 'filled_amount': None, 'avg_price': None}

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancels an active order."""
        if order_id == 'dry-run-limit-buy':
            return True
        try:
            exchange = self.get_exchange()
            exchange.cancel_order(order_id, symbol=symbol)
            print(f"[BinanceClient] 🚫 Canceled unfilled order {order_id} for {symbol}")
            return True
        except Exception as e:
            print(f"[BinanceClient] cancel_order failed for {order_id}: {e}")
            return False

    def place_oco_sell(self, symbol: str, base_amount: float, tp_price: float, sl_price: float) -> Dict[str, Any]:
        """
        Places a native OCO (One-Cancels-the-Other) sell order on Binance.
        """
        exchange = self.get_exchange()
        sl_limit_price = sl_price * 0.998

        if self.dry_run:
            print(f"[BinanceClient] 🧪 DRY_RUN OCO Sell set: {round(base_amount, 6)} {symbol.split('/')[0]}")
            print(f"               Target TP: {tp_price} | Target SL Trigger: {sl_price}")
            return {
                'id': 'dry-run-oco-sell', 'symbol': symbol,
                'filled_amount': round(base_amount, 6),
                'take_profit': tp_price, 'stop_loss': sl_price,
                'dry_run': True, 'status': 'open',
            }

        base_asset = symbol.split('/')[0]
        free_base = self.get_available_balance(base_asset)
        sell_amount = min(base_amount, free_base) if free_base > 0 else base_amount

        market = exchange.market(symbol)
        qty = exchange.amount_to_precision(symbol, sell_amount)
        tp = exchange.price_to_precision(symbol, tp_price)
        sl_tr = exchange.price_to_precision(symbol, sl_price)
        sl_lm = exchange.price_to_precision(symbol, sl_limit_price)

        new_params = {
            'symbol': market['id'],
            'side': 'SELL',
            'quantity': qty,
            'aboveType': 'LIMIT_MAKER',
            'abovePrice': tp,
            'belowType': 'STOP_LOSS_LIMIT',
            'belowStopPrice': sl_tr,
            'belowPrice': sl_lm,
            'belowTimeInForce': 'GTC',
        }
        new_fn = getattr(exchange, 'private_post_orderlist_oco', None) or getattr(exchange, 'privatePostOrderListOco', None)

        if new_fn is not None:
            order = new_fn(new_params)
        else:
            old_fn = getattr(exchange, 'private_post_order_oco', None) or getattr(exchange, 'privatePostOrderOco')
            order = old_fn({
                'symbol': market['id'], 'side': 'SELL', 'quantity': qty,
                'price': tp, 'stopPrice': sl_tr, 'stopLimitPrice': sl_lm,
                'stopLimitTimeInForce': 'GTC',
            })

        print(f"[BinanceClient] 🎯 LIVE OCO Sell set for {symbol}: TP @ {tp} | SL @ {sl_tr} (orderListId: {order.get('orderListId')})")
        return order

    def place_market_buy(self, symbol: str, quote_amount: float, tp_price: Optional[float] = None, sl_price: Optional[float] = None) -> Dict[str, Any]:
        """Direct market buy for instant execution."""
        exchange = self.get_exchange()

        if self.dry_run:
            try:
                ticker = exchange.fetch_ticker(symbol)
                avg_price = float(ticker['last'])
            except Exception:
                avg_price = 100.0
            est_filled = round(quote_amount / avg_price, 6)
            print(f"[BinanceClient] 🧪 DRY_RUN market buy: {quote_amount} USDT of {symbol} (~{est_filled} @ {avg_price:.4f})")
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
        cost = float(order.get('cost', 0) or 0)
        avg_price = (cost / filled) if filled > 0 else float(order.get('price', 0) or 0)
        print(f"[BinanceClient] ✅ LIVE market buy filled: {filled} {symbol.split('/')[0]} @ avg {avg_price:.4f}")
        return {
            'id': order.get('id'), 'symbol': symbol,
            'filled_amount': filled, 'avg_price': avg_price,
            'take_profit': tp_price, 'stop_loss': sl_price,
            'dry_run': False, 'status': order.get('status'),
        }

    def place_market_sell(self, symbol: str, base_amount: float) -> Dict[str, Any]:
        """Direct market sell for immediate position liquidation."""
        exchange = self.get_exchange()
        formatted_amount = float(exchange.amount_to_precision(symbol, base_amount)) if not self.dry_run else round(base_amount, 6)

        if self.dry_run:
            try:
                ticker = exchange.fetch_ticker(symbol)
                avg_price = float(ticker['last'])
            except Exception:
                avg_price = 100.0
            print(f"[BinanceClient] 🧪 DRY_RUN market sell: {formatted_amount} {symbol.split('/')[0]}")
            return {
                'id': 'dry-run-market-sell', 'symbol': symbol,
                'filled_amount': formatted_amount, 'avg_price': avg_price,
                'dry_run': True, 'status': 'closed',
            }

        order = exchange.create_order(
            symbol=symbol, type='market', side='sell', amount=formatted_amount,
        )
        filled = float(order.get('filled', 0) or 0)
        cost = float(order.get('cost', 0) or 0)
        avg_price = (cost / filled) if filled > 0 else float(order.get('price', 0) or 0)
        print(f"[BinanceClient] ✅ LIVE market sell filled: {filled} {symbol.split('/')[0]} @ avg {avg_price:.4f}")
        return {
            'id': order.get('id'), 'symbol': symbol,
            'filled_amount': filled, 'avg_price': avg_price,
            'dry_run': False, 'status': order.get('status'),
        }


# ── Global Default Client & Functional API ───────────────────────
_default_client: Optional[BinanceClient] = None

def get_binance_client() -> BinanceClient:
    global _default_client
    if _default_client is None:
        _default_client = BinanceClient()
    return _default_client

def fetch_ohlcv(symbol: str = 'SOL/USDT', timeframe: str = '5m', limit: int = 1000) -> pd.DataFrame:
    return get_binance_client().fetch_ohlcv(symbol, timeframe, limit)

def fetch_current_price(symbol: str = 'SOL/USDT') -> float:
    return get_binance_client().fetch_current_price(symbol)

def get_available_balance(asset: str) -> float:
    return get_binance_client().get_available_balance(asset)

def place_limit_buy(symbol: str, limit_price: float, quote_amount: float) -> Dict[str, Any]:
    return get_binance_client().place_limit_buy(symbol, limit_price, quote_amount)

def check_order_status(symbol: str, order_id: str) -> Dict[str, Any]:
    return get_binance_client().check_order_status(symbol, order_id)

def cancel_order(symbol: str, order_id: str) -> bool:
    return get_binance_client().cancel_order(symbol, order_id)

def place_oco_sell(symbol: str, base_amount: float, tp_price: float, sl_price: float) -> Dict[str, Any]:
    return get_binance_client().place_oco_sell(symbol, base_amount, tp_price, sl_price)

def place_market_buy(symbol: str, quote_amount: float, tp_price: Optional[float] = None, sl_price: Optional[float] = None) -> Dict[str, Any]:
    return get_binance_client().place_market_buy(symbol, quote_amount, tp_price, sl_price)

def place_market_sell(symbol: str, base_amount: float) -> Dict[str, Any]:
    return get_binance_client().place_market_sell(symbol, base_amount)

