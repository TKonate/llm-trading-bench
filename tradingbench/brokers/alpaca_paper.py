"""Alpaca paper-trading broker (REST, no SDK).

Why raw REST instead of `alpaca-py`: the SDK pulls a large dependency tree for
what is, in our case, six endpoints. Talking HTTP directly keeps the mapping
explicit — and explicit mapping is exactly what we will need when we add a live
broker in phase B.

Paper account keys are free and require no KYC: https://app.alpaca.markets
Set ALPACA_API_KEY / ALPACA_SECRET_KEY in `.env`.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import httpx

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

TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


def _dt(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


_TIMEFRAME = re.compile(r"^(\d+)(Min|Hour|Day|Week|Month)$")


def _lookback_start(timeframe: str, limit: int, now: datetime | None = None) -> datetime:
    """Fenêtre de recherche assez large pour contenir `limit` barres utiles.

    Facteur 2 + marge : un marché actions ne travaille qu'environ 5 jours sur 7,
    et il y a les jours fériés. Trop court = moins de barres que demandé.
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

    # -- plumbing --------------------------------------------------------
    def _request(self, method: str, url: str, **kwargs) -> dict | list:
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

    # -- market data -----------------------------------------------------
    def get_quote(self, symbol: str) -> Quote:
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
        # Contraintes réelles de l'API Alpaca, vérifiées sur le compte paper :
        #   - sans `start`, la réponse est 200 avec `bars: null` (pas d'erreur) ;
        #   - avec `sort=asc` + `start`, on obtient les PLUS ANCIENNES barres de la fenêtre.
        # On demande donc les plus récentes (desc) sur une fenêtre large, puis on
        # remet dans l'ordre chronologique — c'est le contrat du Broker.
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

    # -- account ---------------------------------------------------------
    def get_account(self) -> Account:
        payload = self._request("GET", f"{self._trading_url}/v2/account")
        return Account(
            account_id=str(payload.get("account_number") or payload.get("id") or "unknown"),
            cash=float(payload.get("cash") or 0),
            equity=float(payload.get("equity") or 0),
            buying_power=float(payload.get("buying_power") or 0),
            currency=str(payload.get("currency") or "USD"),
        )

    def get_positions(self) -> list[Position]:
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

    # -- execution -------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: Side,
        *,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        client_order_id: str | None = None,
    ) -> Order:
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
        if order_type is OrderType.LIMIT:
            if limit_price is None:
                raise BrokerError("limit orders require limit_price")
            body["limit_price"] = str(limit_price)
        payload = self._request("POST", f"{self._trading_url}/v2/orders", json=body)
        return self._to_order(payload)

    def get_order(self, order_id: str) -> Order:
        return self._to_order(self._request("GET", f"{self._trading_url}/v2/orders/{order_id}"))

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"{self._trading_url}/v2/orders/{order_id}")

    def is_market_open(self) -> bool:
        payload = self._request("GET", f"{self._trading_url}/v2/clock")
        return bool(payload.get("is_open"))

    # -- mapping ---------------------------------------------------------
    @staticmethod
    def _to_order(payload: dict) -> Order:
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
