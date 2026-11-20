"""Shared domain types: the contract between the decision loop and any broker.

Plain frozen dataclasses on purpose: cheap to construct, trivially serialisable
to JSONL, and immutable so that a quote or an order threaded through the agent
loop cannot be edited in place and silently disagree with what the broker
returned. They carry only the arithmetic both sides must agree on (market
value, unrealised P&L, spread in basis points).

No rounding lives here: rounding is an execution concern and belongs to the
broker that has to honour a price grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    """Order direction. Lowercase values are what brokers expect on the wire."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Only the two types the harness can fill honestly."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Lifecycle states we can act on. Values match Alpaca's vocabulary."""

    NEW = "new"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, raw: str | None) -> OrderStatus:
        """Map a broker status string, degrading to UNKNOWN instead of raising.

        Brokers add statuses without warning: Alpaca has added `pending_new`,
        `replaced` and `held` over time. A reconciliation loop that crashes on
        the first unknown string stops measuring; one that sees UNKNOWN can
        still report the order and keep going.
        """
        try:
            return cls((raw or "").lower())
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True, slots=True)
class Account:
    """Snapshot of the account. `buying_power` is broker-defined, not cash."""

    account_id: str
    cash: float
    equity: float
    buying_power: float
    currency: str = "USD"


@dataclass(frozen=True, slots=True)
class Position:
    """One open holding.

    `current_price` is optional because a position can be read from a broker
    that does not mark to market; `last_price` then falls back to the entry
    price, which reports zero unrealised P&L rather than a phantom gain.
    """

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float | None = None

    @property
    def last_price(self) -> float:
        return self.current_price if self.current_price is not None else self.avg_entry_price

    @property
    def market_value(self) -> float:
        """Signed value: a short position (negative qty) counts against equity."""
        return self.qty * self.last_price

    @property
    def unrealized_pl(self) -> float:
        return self.qty * (self.last_price - self.avg_entry_price)


@dataclass(frozen=True, slots=True)
class Quote:
    """Top of book. No size, because every broker reports it differently."""

    symbol: str
    bid: float
    ask: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        # A stale quote outside trading hours routinely comes back as 0/0; this
        # guard keeps the caller from dividing by zero and turns the case into
        # "no spread information", which the agent can act on.
        mid = self.mid
        return 0.0 if mid == 0 else (self.ask - self.bid) / mid * 10_000


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV candle. `timestamp` is the bar's opening time, UTC."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class NewsItem:
    """Untrusted external text: treat it as data, never as an instruction."""

    timestamp: datetime
    headline: str
    summary: str = ""
    symbols: tuple[str, ...] = ()
    source: str = ""
    url: str = ""

    def one_line(self, width: int = 160) -> str:
        """Single-line rendering for a prompt.

        Newlines are flattened because a multi-line headline would break the
        bullet list the model reads; `width` truncation keeps one wire story
        from crowding out the rest of the context.
        """
        head = self.headline.strip().replace("\n", " ")
        if len(head) > width:
            head = head[: width - 1] + "…"
        return f"{self.timestamp.date().isoformat()} · {head}"


@dataclass(frozen=True, slots=True)
class Order:
    """Broker view of an order.

    `filled_qty` and `filled_avg_price` are kept separate from `qty` because a
    partial fill is a normal outcome: `qty` is what was asked, the fill fields
    are what happened, and collapsing the two would make a half execution look
    like a full one.
    """

    id: str
    symbol: str
    qty: float
    side: Side
    order_type: OrderType
    status: OrderStatus
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    submitted_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        """True once the order can no longer change.

        PARTIALLY_FILLED is deliberately absent: the order is still live and
        cancelling it would leave a half position behind.
        """
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        }
