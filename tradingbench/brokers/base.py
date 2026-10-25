"""Broker abstraction (project decision D5).

The decision loop must never import Alpaca. Swapping paper for live in phase B
then costs one new subclass instead of a rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Account, Bar, Order, OrderType, Position, Quote, Side


class BrokerError(RuntimeError):
    """Raised by any execution backend when a call cannot be completed."""


class Broker(ABC):
    """Minimal surface needed by an LLM agent to trade and read the market."""

    name: str = "abstract"
    is_paper: bool = True

    @abstractmethod
    def get_account(self) -> Account:
        """Cash, equity and buying power of the account behind this broker."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Open positions, empty list when flat."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Latest bid/ask for one symbol."""

    @abstractmethod
    def get_bars(
        self, symbol: str, *, timeframe: str = "1Day", limit: int = 30
    ) -> list[Bar]:
        """Most recent bars, oldest first."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: Side,
        *,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        """Submit an order. Market orders fill immediately in paper."""

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """Current state of one order."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order (no-op if already terminal)."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """Whether the exchange is currently accepting orders."""
