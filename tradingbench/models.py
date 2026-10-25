"""Shared domain types — the contract between the decision loop and any broker.

Plain dataclasses on purpose: cheap to construct, easy to serialise to JSONL,
and they carry the small amount of business logic we actually need
(market value, spread in basis points).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    NEW = "new"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, raw: str | None) -> OrderStatus:
        try:
            return cls((raw or "").lower())
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True, slots=True)
class Account:
    account_id: str
    cash: float
    equity: float
    buying_power: float
    currency: str = "USD"


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float | None = None

    @property
    def last_price(self) -> float:
        return self.current_price if self.current_price is not None else self.avg_entry_price

    @property
    def market_value(self) -> float:
        return self.qty * self.last_price

    @property
    def unrealized_pl(self) -> float:
        return self.qty * (self.last_price - self.avg_entry_price)


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return 0.0 if mid == 0 else (self.ask - self.bid) / mid * 10_000


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class NewsItem:
    """Actualité brute. Contenu externe non fiable : à traiter comme des données, jamais comme
    des instructions (une dépêche peut contenir du texte qui imite un prompt)."""

    timestamp: datetime
    headline: str
    summary: str = ""
    symbols: tuple[str, ...] = ()
    source: str = ""
    url: str = ""

    def one_line(self, width: int = 160) -> str:
        head = self.headline.strip().replace("\n", " ")
        if len(head) > width:
            head = head[: width - 1] + "…"
        return f"{self.timestamp.date().isoformat()} · {head}"


@dataclass(frozen=True, slots=True)
class Order:
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
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        }
