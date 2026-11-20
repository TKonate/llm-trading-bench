"""Alpaca paper-trading broker over raw REST, without the SDK.

Why raw REST instead of `alpaca-py`: the SDK pulls a large dependency tree for
what is, in our case, six endpoints. Talking HTTP directly keeps the mapping
explicit, and explicit mapping is exactly what we will need when a live broker
arrives in phase B.

Paper account keys are free and require no KYC: https://app.alpaca.markets.
Set ALPACA_API_KEY / ALPACA_SECRET_KEY in `.env`.

Deliberately no retry, no websocket and no order streaming: a retried submit is
a duplicate order, and a benchmark that silently retries hides exactly the
failures it is supposed to measure.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import httpx

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

TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


def _dt(raw: str | None) -> datetime:
    """Parse an Alpaca timestamp, defaulting to now when the field is absent.

    Alpaca writes `2026-01-05T14:30:00Z`; `fromisoformat` only learned to read a
    trailing `Z` in 3.11, and this package supports 3.10, so the Z is rewritten.
    """
    if not raw:
        return datetime.now(UTC)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


_TIMEFRAME = re.compile(r"^(\d+)(Min|Hour|Day|Week|Month)$")


def _lookback_start(timeframe: str, limit: int, now: datetime | None = None) -> datetime:
    """Start of a window wide enough to contain `limit` usable bars.

    Factor two plus a margin: an equity market works about five days out of
    seven, and holidays remove more. Too narrow a window returns fewer bars than
    asked for, which then looks like a data outage. Month is treated as 30 days:
    the only rounded unit in the table.
    """
    now = now or datetime.now(UTC)
    match = _TIMEFRAME.match(timeframe)
    if not match:
        return now - timedelta(days=max(30, limit * 2 + 10))
    step, unit = int(match.group(1)), match.group(2)
    unit_delta = {
        "Min": timedelta(minutes=step),
        "Hour": timedelta(hours=step),
        "Day": timedelta(days=step),
        "Week": timedelta(weeks=step),
        "Month": timedelta(days=30 * step),
    }[unit]
    return now - unit_delta * limit * 2 - timedelta(days=5)


class AlpacaPaperBroker(Broker):
    name = "alpaca-paper"
    is_paper = True

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        feed: str = "iex",
        trading_url: str = TRADING_URL,
        data_url: str = DATA_URL,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key or not secret_key:
            raise BrokerError("Alpaca credentials missing (ALPACA_API_KEY / ALPACA_SECRET_KEY)")
        self._feed = feed
        self._trading_url = trading_url.rstrip("/")
        self._data_url = data_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "accept": "application/json",
            },
        )

    def _request(self, method: str, url: str, **kwargs) -> dict | list:
        """One HTTP call, with the two error shapes the callers must not see.

        A transport failure and an HTTP error both become BrokerError, because
        the executor turns that into a tool result the model can read. An empty
        body returns `{}`: DELETE /v2/orders answers 204 with no content, and
        treating that as a parse error would report a successful cancel as a
        failure.
        """
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:  # network layer
            raise BrokerError(f"{method} {url} failed: {exc}") from exc
        if response.status_code >= 400:
            raise BrokerError(
                f"{method} {url} -> HTTP {response.status_code}: {response.text[:300]}"
            )
        if not response.content:
            return {}
        return response.json()

    def get_quote(self, symbol: str) -> Quote:
        """Latest quote from the configured feed.

        `feed` defaults to `iex` because the free paper plan has no SIP access:
        asking for SIP is rejected with a subscription error rather than served
        as a delayed quote, so defaulting beats failing deep in a session.
        """
        payload = self._request(
            "GET",
            f"{self._data_url}/v2/stocks/{symbol}/quotes/latest",
            params={"feed": self._feed},
        )
        quote = payload.get("quote") or {}
        if not quote:
            raise BrokerError(f"no quote returned for {symbol} (feed={self._feed})")
        return Quote(
            symbol=symbol,
            bid=float(quote["bp"]),
            ask=float(quote["ap"]),
            timestamp=_dt(quote.get("t")),
        )

    def get_bars(self, symbol: str, *, timeframe: str = "1Day", limit: int = 30) -> list[Bar]:
        """Recent bars, oldest first, unadjusted.

        Two API constraints, checked against a live paper account, drive the
        shape of this call: without `start` the answer is a 200 whose `bars`
        field is null, and with `sort=asc` the API returns the *oldest* bars of
        the window. So the request asks for the newest ones (desc) over a wide
        window, and the list is reversed here to honour the Broker contract.

        `adjustment` is raw: a fill is journaled at the price printed that day,
        and a split- or dividend-adjusted history would not line up with it.
        """
        payload = self._request(
            "GET",
            f"{self._data_url}/v2/stocks/{symbol}/bars",
            params={
                "timeframe": timeframe,
                "limit": limit,
                "feed": self._feed,
                "adjustment": "raw",
                "sort": "desc",
                "start": _lookback_start(timeframe, limit).isoformat(),
            },
        )
        rows = payload.get("bars") or []
        return [
            Bar(
                symbol=symbol,
                timestamp=_dt(row.get("t")),
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row["v"]),
            )
            for row in reversed(rows)
        ]

    def get_news(self, symbols: list[str], *, limit: int = 5) -> list[NewsItem]:
        """Headlines for these symbols, from the v1beta1 news endpoint.

        `exclude_contentless` is set because a headline with no summary gives
        the agent a date and a title it cannot act on. The payload is checked for
        a dict before `.get` is called, because `_request` can legitimately
        return a list.
        """
        if not symbols:
            return []
        payload = self._request(
            "GET",
            f"{self._data_url}/v1beta1/news",
            params={
                "symbols": ",".join(symbols),
                "limit": limit,
                "exclude_contentless": "true",
            },
        )
        rows = payload.get("news") if isinstance(payload, dict) else None
        return [
            NewsItem(
                timestamp=_dt(row.get("created_at")),
                headline=str(row.get("headline") or ""),
                summary=str(row.get("summary") or ""),
                symbols=tuple(str(s) for s in row.get("symbols") or ()),
                source=str(row.get("source") or ""),
                url=str(row.get("url") or ""),
            )
            for row in rows or []
        ]

    def get_account(self) -> Account:
        """Account snapshot. Every money field arrives as a string and is cast here.

        `account_number` is the human-readable identifier; `id` is the UUID. The
        fallback exists because older accounts only carry one of the two.
        """
        payload = self._request("GET", f"{self._trading_url}/v2/account")
        return Account(
            account_id=str(payload.get("account_number") or payload.get("id") or "unknown"),
            cash=float(payload.get("cash") or 0),
            equity=float(payload.get("equity") or 0),
            buying_power=float(payload.get("buying_power") or 0),
            currency=str(payload.get("currency") or "USD"),
        )

    def get_positions(self) -> list[Position]:
        """Open positions. The endpoint answers with a JSON list, not an object."""
        payload = self._request("GET", f"{self._trading_url}/v2/positions")
        rows = payload if isinstance(payload, list) else []
        return [
            Position(
                symbol=str(row["symbol"]),
                qty=float(row["qty"]),
                avg_entry_price=float(row["avg_entry_price"]),
                current_price=float(row["current_price"]) if row.get("current_price") else None,
            )
            for row in rows
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
        """Submit an order and return the broker's view of it.

        `qty` is sent as text because a quantity computed in the agent loop can
        carry float noise (0.30000000000000004), and the API expects a decimal
        string for fractional orders. Protective leg prices are rounded to two
        decimals for the same reason: that is the precision the API expects on
        a stop or a limit leg.

        `time_in_force` is day: bracket and OTO orders only accept day, and a
        GTC entry would outlive the experiment that placed it.
        """
        if qty <= 0:
            raise BrokerError("qty must be positive")
        body: dict[str, object] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.value,
            "type": order_type.value,
            "time_in_force": "day",
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        if stop_loss is not None or take_profit is not None:
            # Alpaca: bracket = TP and SL, oto = a single protection. On a sale
            # the protection makes no sense; it would need a separate buy-back
            # order, which is not modelled.
            if side is Side.SELL:
                raise BrokerError("stop_loss / take_profit ne s'appliquent qu'à un achat")
            body["order_class"] = "bracket" if (stop_loss and take_profit) else "oto"
            if take_profit is not None:
                body["take_profit"] = {"limit_price": f"{round(float(take_profit), 2)}"}
            if stop_loss is not None:
                body["stop_loss"] = {"stop_price": f"{round(float(stop_loss), 2)}"}
        if order_type is OrderType.LIMIT:
            if limit_price is None:
                raise BrokerError("limit orders require limit_price")
            body["limit_price"] = str(limit_price)
        payload = self._request("POST", f"{self._trading_url}/v2/orders", json=body)
        return self._to_order(payload)

    def get_order(self, order_id: str) -> Order:
        return self._to_order(self._request("GET", f"{self._trading_url}/v2/orders/{order_id}"))

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order. The API answers 204 with no body, so nothing to map."""
        self._request("DELETE", f"{self._trading_url}/v2/orders/{order_id}")

    def is_market_open(self) -> bool:
        """Ask the clock endpoint rather than computing a calendar locally.

        A local calendar would need every holiday rule, and would be wrong on
        the first unannounced half session. The cost is one HTTP call per
        session, which the caller pays anyway.
        """
        payload = self._request("GET", f"{self._trading_url}/v2/clock")
        return bool(payload.get("is_open"))

    @staticmethod
    def _to_order(payload: dict) -> Order:
        """Map the API's order object onto our Order.

        `filled_avg_price` is null until the first fill, and an empty string on
        some cancelled orders; both become None, which is what "no execution
        price yet" means further up. Required fields are read with `[]` so a
        shape change fails loudly instead of producing an order with no symbol.
        """
        filled_avg = payload.get("filled_avg_price")
        return Order(
            id=str(payload["id"]),
            symbol=str(payload["symbol"]),
            qty=float(payload.get("qty") or 0),
            side=Side(str(payload["side"]).lower()),
            order_type=OrderType(str(payload["type"]).lower()),
            status=OrderStatus.parse(payload.get("status")),
            filled_qty=float(payload.get("filled_qty") or 0),
            filled_avg_price=float(filled_avg) if filled_avg else None,
            submitted_at=_dt(payload.get("submitted_at")),
        )

    def close(self) -> None:
        self._client.close()
