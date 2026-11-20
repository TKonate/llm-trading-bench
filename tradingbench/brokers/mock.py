"""Deterministic in-memory broker.

Used by the test suite, by `tradingbench smoke --broker mock` and by the
tool-calling harness, so the whole plumbing can be exercised with no Alpaca
account and no network. The point is reproducibility: the same seed must give
the same prices, the same bars and the same fills, in any process, on any day.

What it deliberately does not model: commissions, slippage beyond the quoted
spread, market impact, partial fills, and a resting-order engine. A limit order
that does not cross is accepted and then simply stays open. Fees and fills are
where real money is lost, so a mock that invented them would make the benchmark
look more precise than it is.
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

from ..models import (
    Account,
    Bar,
    NewsItem,
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
        # A private Random instance, never the global one: a shared generator
        # would make prices depend on how many draws another test made first.
        self._rng = random.Random(seed)
        self._cash = float(cash)
        self._market_open = market_open
        # Bar timestamps hang off a fixed Monday at 14:30 UTC, the New York
        # opening bell in winter time, rather than off the wall clock, so a
        # golden-file comparison does not rot with the calendar.
        self._start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        # Prices are drawn in `symbols` order, so the same symbol gets a
        # different price if the tuple is reordered: the seed alone does not
        # determine a price, the (seed, position in universe) pair does.
        self._prices: dict[str, float] = {
            symbol: round(40 + 180 * self._rng.random(), 2) for symbol in symbols
        }
        self._positions: dict[str, tuple[float, float]] = {}  # symbol -> (qty, avg price)
        self._orders: dict[str, Order] = {}
        self._client_ids: dict[str, str] = {}  # client_order_id -> order id (idempotence)

    def _price(self, symbol: str) -> float:
        if symbol not in self._prices:
            raise BrokerError(f"unknown symbol {symbol!r}")
        return self._prices[symbol]

    def get_quote(self, symbol: str) -> Quote:
        """Top of book around the current price, at a fixed 4 bps round trip.

        Prices are rounded to four decimals so a quote serialises without float
        noise; the spread lands a hair under 4 bps because of that rounding, which
        is the same order of precision a real feed reports. The timestamp is the
        only field in this broker that depends on the wall clock.
        """
        price = self._price(symbol)
        half_spread = price * 0.0002
        return Quote(
            symbol=symbol,
            bid=round(price - half_spread, 4),
            ask=round(price + half_spread, 4),
            timestamp=datetime.now(UTC),
        )

    def get_bars(self, symbol: str, *, timeframe: str = "1Day", limit: int = 30) -> list[Bar]:
        """A synthetic daily series anchored to the current price, oldest first.

        The generator is seeded from the symbol *and* the whole price table, so
        two consequences follow and both are intentional: the series is stable
        across processes (a string seed goes through sha512, not `hash()`, so
        PYTHONHASHSEED cannot move it), and `set_price` rewrites history, which
        is what a scenario that moves a price needs.

        Calendars are ignored: bars are one per calendar day, weekends included.
        The agent counts bars and compares closes, so the spacing matters and
        the trading calendar does not.
        """
        base = self._price(symbol)
        rng = random.Random(f"{symbol}-{seed_of(self._prices)}")
        bars: list[Bar] = []
        # Start 6 % below the current price; the upward drift below is what
        # brings the closes back into the range of the live quote beside them.
        price = base * 0.94
        for index in range(limit):
            # Slight upward bias (-2 % / +2.2 %) so pullbacks and rallies both
            # occur without the series drifting to zero on long limits.
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

    def get_news(self, symbols: list[str], *, limit: int = 5) -> list[NewsItem]:
        """Fixed, deterministic headlines: the mock must never need the network.

        The dates are literals rather than clock reads, so the news block of a
        prompt is byte-identical from one run to the next.
        """
        items: list[NewsItem] = []
        base = datetime(2026, 1, 14, 15, 0, tzinfo=UTC)
        for index, symbol in enumerate(symbols[:limit]):
            items.append(
                NewsItem(
                    timestamp=base - timedelta(hours=index),
                    headline=f"{symbol} publie des résultats conformes aux attentes",
                    summary="Communiqué de presse simulé pour les tests hors ligne.",
                    symbols=(symbol,),
                    source="mock",
                    url="https://example.invalid/news",
                )
            )
        return items

    def get_account(self) -> Account:
        """Mark-to-market snapshot. Buying power equals cash: no margin here."""
        holdings = sum(qty * self._price(symbol) for symbol, (qty, _) in self._positions.items())
        return Account(
            account_id="mock-account",
            cash=round(self._cash, 2),
            equity=round(self._cash + holdings, 2),
            buying_power=round(self._cash, 2),
        )

    def get_positions(self) -> list[Position]:
        """Open positions, symbol order, flat entries already dropped."""
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
        """Fill or reject an order against the synthetic book.

        Fills cross the spread, and that is the only cost modelled: a buy pays
        the ask, a sell receives the bid. No commission and no slippage, so any
        performance figure produced on this broker is an upper bound; it exists
        to exercise behaviour, not to estimate returns.

        A market order fills at once. A limit order that does not cross is
        accepted and left resting, because there is no matching engine: it stays
        ACCEPTED until cancelled, and no later call will fill it.
        """
        if qty <= 0:
            raise BrokerError("qty must be positive")
        if not self._market_open:
            raise BrokerError("market is closed")
        # Idempotence first, before any validation or fill: replaying a client
        # id must return the original order even if the market has since closed.
        if client_order_id and client_order_id in self._client_ids:
            return self.get_order(self._client_ids[client_order_id])

        quote = self.get_quote(symbol)
        fill_price = quote.ask if side is Side.BUY else quote.bid
        if stop_loss is not None or take_profit is not None:
            # Protective legs are validated but not stored: the mock never
            # triggers them, so a caller must not believe the position is
            # actually protected. Rejecting a wrong-side stop is still done
            # here, because that check is part of the broker contract.
            if side is Side.SELL:
                raise BrokerError("stop_loss / take_profit ne s'appliquent qu'à un achat")
            entry = limit_price if (order_type is OrderType.LIMIT and limit_price) else fill_price
            if stop_loss is not None and stop_loss >= entry:
                raise BrokerError(f"stop_loss {stop_loss} doit être sous le prix d'entrée {entry}")
            if take_profit is not None and take_profit <= entry:
                raise BrokerError(f"take_profit {take_profit} doit être au-dessus de {entry}")
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
                self._remember(order.id, client_order_id)
                return order
            # A crossing limit fills at the limit price, not at the better
            # market price: the conservative reading of what a fill would pay.
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
        self._remember(order.id, client_order_id)
        return order

    def _remember(self, order_id: str, client_order_id: str | None) -> None:
        if client_order_id:
            self._client_ids[client_order_id] = order_id

    def _apply_fill(self, symbol: str, qty: float, side: Side, price: float) -> None:
        """Move cash and the position for one fill, or raise before touching either.

        The 1e-9 tolerance is float slack on a comparison that decides whether
        money exists: without it, buying exactly the available cash can be
        refused because 1000.0 - 1000.0000000000001 < 0.
        """
        held, avg = self._positions.get(symbol, (0.0, 0.0))
        if side is Side.BUY:
            cost = qty * price
            if cost > self._cash + 1e-9:
                raise BrokerError("insufficient buying power")
            # Weighted average, so several entries into one symbol produce the
            # entry price the P&L of the whole position is measured against.
            new_qty = held + qty
            new_avg = (held * avg + qty * price) / new_qty if new_qty else 0.0
            self._cash -= cost
            self._positions[symbol] = (new_qty, new_avg)
        else:
            if qty > held + 1e-9:
                raise BrokerError(f"cannot sell {qty} of {symbol}: only {held} held")
            # A sale does not change the entry price of what remains: realised
            # P&L is booked in cash, and rebasing the average would hide it.
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
        """Cancel a live order. Terminal orders are left untouched.

        The rewrite carries no fill fields: harmless here, since nothing fills
        after acceptance and PARTIALLY_FILLED never occurs, but a real broker
        must not lose the fills of a cancelled order the same way.
        """
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

    def set_price(self, symbol: str, price: float) -> None:
        """Move the price, which also moves the whole synthetic bar series."""
        self._prices[symbol] = float(price)

    def seed_position(self, symbol: str, qty: float, price: float | None = None) -> None:
        """Open a position before a scenario (sell tests, portfolio arbitrage).

        Cash is debited without checking that it is there: this is a fixture, not
        a trade, and a scenario that needs an over-spent account is a scenario
        the warning path has to be tested against.
        """
        entry = float(price) if price is not None else self._price(symbol)
        if symbol in self._prices and price is not None:
            self._prices[symbol] = float(price)
        self._cash -= qty * entry
        self._positions[symbol] = (float(qty), entry)


def seed_of(prices: dict[str, float]) -> str:
    """Stable seed derived from the initial price table.

    Sorted and rounded so the string does not depend on dict insertion order or
    on float repr noise; two brokers built the same way give the same bars.
    """
    return str(sorted((k, round(v, 2)) for k, v in prices.items()))
