"""Broker abstraction (project decision D5).

The decision loop must never import Alpaca. Swapping paper for live in phase B
then costs one new subclass instead of a rewrite.

The surface below is the whole contract: market data in, orders out. Anything
that needs a venue-specific feature (bracket legs, feed selection, clock
calendars) stays inside a subclass, so no caller can depend on it. Quantities
are floats because fractional shares are allowed, and are often the only way to
stay under a 25 % position limit when one share costs more than that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Account, Bar, NewsItem, Order, OrderType, Position, Quote, Side


class BrokerError(RuntimeError):
    """Raised by any execution backend when a call cannot be completed.

    One exception type for every failure mode: the executor turns it into a
    tool result the model can read, and a taxonomy would only add branches
    without changing what the agent does next.
    """


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
        """Most recent bars, oldest first.

        The chronological order is part of the contract, not a convenience: the
        agent reasons about a series, and a reversed list silently inverts every
        trend it derives. Implementations sort, even when the API answers
        newest-first.
        """

    @abstractmethod
    def get_news(self, symbols: list[str], *, limit: int = 5) -> list[NewsItem]:
        """Latest news for these symbols; an empty list is a valid answer."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: Side,
        *,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit an order. Market orders fill immediately in paper.

        `stop_loss` / `take_profit` create a protective order alongside the entry
        (bracket when both are given). They only make sense on a buy entry.

        `client_order_id` must be idempotent: replaying the same id returns the
        original order instead of placing a second one. That is the last line of
        defence against double orders after a crash between "order sent" and
        "state saved".
        """

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """Current state of one order."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order (no-op if already terminal)."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """Whether the exchange is currently accepting orders."""
