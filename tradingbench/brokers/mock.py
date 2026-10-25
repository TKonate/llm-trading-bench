"""Deterministic in-memory broker.

Used by the test suite and by `tradingbench smoke --broker mock`, so the whole
plumbing can be exercised without an Alpaca account or network access.
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

from ..models import (
    Account,
    Bar,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
)
from .base import Broker, BrokerError

DEFAULT_SYMBOLS = ("AAPL", "MSFT", "NVDA", "SPY")


class MockBroker(Broker):
    name = "mock"
    is_paper = True

    def __init__(
        self,
        symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
        cash: float = 10_000.0,
        seed: int = 7,
        market_open: bool = True,
    ) -> None:
        self._rng = random.Random(seed)
        self._cash = float(cash)
        self._market_open = market_open
        self._start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        self._prices: dict[str, float] = {
            symbol: round(40 + 180 * self._rng.random(), 2) for symbol in symbols
        }
        self._positions: dict[str, tuple[float, float]] = {}  # symbol -> (qty, avg price)
        self._orders: dict[str, Order] = {}

    # -- market data -----------------------------------------------------
    def _price(self, symbol: str) -> float:
        if symbol not in self._prices:
            raise BrokerError(f"unknown symbol {symbol!r}")
        return self._prices[symbol]

    def get_quote(self, symbol: str) -> Quote:
        price = self._price(symbol)
        half_spread = price * 0.0002
        return Quote(
            symbol=symbol,
            bid=round(price - half_spread, 4),
            ask=round(price + half_spread, 4),
            timestamp=datetime.now(UTC),
        )

    def get_bars(self, symbol: str, *, timeframe: str = "1Day", limit: int = 30) -> list[Bar]:
        base = self._price(symbol)
        rng = random.Random(f"{symbol}-{seed_of(self._prices)}")
        bars: list[Bar] = []
        price = base * 0.94
        for index in range(limit):
            drift = rng.uniform(-0.02, 0.022)
            open_ = round(price, 2)
            close = round(max(1.0, price * (1 + drift)), 2)
            high = round(max(open_, close) * (1 + abs(rng.uniform(0, 0.006))), 2)
            low = round(min(open_, close) * (1 - abs(rng.uniform(0, 0.006))), 2)
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=self._start + timedelta(days=index),
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=float(rng.randint(1_000_000, 40_000_000)),
                )
            )
            price = close
        return bars

    # -- account ---------------------------------------------------------
    def get_account(self) -> Account:
        holdings = sum(qty * self._price(symbol) for symbol, (qty, _) in self._positions.items())
        return Account(
            account_id="mock-account",
            cash=round(self._cash, 2),
            equity=round(self._cash + holdings, 2),
            buying_power=round(self._cash, 2),
        )

    def get_positions(self) -> list[Position]:
        return [
            Position(
                symbol=symbol,
                qty=qty,
                avg_entry_price=avg,
                current_price=self._price(symbol),
            )
            for symbol, (qty, avg) in sorted(self._positions.items())
            if qty != 0
        ]

    # -- execution -------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: Side,
        *,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        if qty <= 0:
            raise BrokerError("qty must be positive")
        if not self._market_open:
            raise BrokerError("market is closed")

        quote = self.get_quote(symbol)
        fill_price = quote.ask if side is Side.BUY else quote.bid
        if order_type is OrderType.LIMIT:
            if limit_price is None:
                raise BrokerError("limit orders require limit_price")
            crosses = limit_price >= quote.ask if side is Side.BUY else limit_price <= quote.bid
            if not crosses:
                order = Order(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type=order_type,
                    status=OrderStatus.ACCEPTED,
                    submitted_at=datetime.now(UTC),
                )
                self._orders[order.id] = order
                return order
            fill_price = limit_price

        self._apply_fill(symbol, qty, side, fill_price)
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            status=OrderStatus.FILLED,
            filled_qty=qty,
            filled_avg_price=fill_price,
            submitted_at=datetime.now(UTC),
        )
        self._orders[order.id] = order
        return order

    def _apply_fill(self, symbol: str, qty: float, side: Side, price: float) -> None:
        held, avg = self._positions.get(symbol, (0.0, 0.0))
        if side is Side.BUY:
            cost = qty * price
            if cost > self._cash + 1e-9:
                raise BrokerError("insufficient buying power")
            new_qty = held + qty
            new_avg = (held * avg + qty * price) / new_qty if new_qty else 0.0
            self._cash -= cost
            self._positions[symbol] = (new_qty, new_avg)
        else:
            if qty > held + 1e-9:
                raise BrokerError(f"cannot sell {qty} of {symbol}: only {held} held")
            self._cash += qty * price
            remaining = held - qty
            if remaining == 0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = (remaining, avg)

    def get_order(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise BrokerError(f"unknown order {order_id!r}") from exc

    def cancel_order(self, order_id: str) -> None:
        order = self.get_order(order_id)
        if order.is_terminal:
            return
        self._orders[order_id] = Order(
            id=order.id,
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            order_type=order.order_type,
            status=OrderStatus.CANCELED,
            submitted_at=order.submitted_at,
        )

    def is_market_open(self) -> bool:
        return self._market_open

    # -- test helpers ----------------------------------------------------
    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = float(price)

    def seed_position(self, symbol: str, qty: float, price: float | None = None) -> None:
        """Ouvre une position d'avance (scénarios de vente, tests de portefeuille)."""
        entry = float(price) if price is not None else self._price(symbol)
        if symbol in self._prices and price is not None:
            self._prices[symbol] = float(price)
        self._cash -= qty * entry
        self._positions[symbol] = (float(qty), entry)


def seed_of(prices: dict[str, float]) -> str:
    """Stable seed derived from the initial price table."""
    return str(sorted((k, round(v, 2)) for k, v in prices.items()))
