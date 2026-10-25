"""Mapping tests: Alpaca JSON -> domain objects, and the failure path."""

from __future__ import annotations

import httpx
import pytest

from tradingbench.brokers.alpaca_paper import AlpacaPaperBroker
from tradingbench.brokers.base import BrokerError
from tradingbench.models import OrderStatus


def _broker(handler) -> AlpacaPaperBroker:
    return AlpacaPaperBroker(
        api_key="k", secret_key="s", transport=httpx.MockTransport(handler), feed="iex"
    )


def test_feed_is_forwarded_to_the_data_api():
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(
            200, json={"quote": {"bp": 1.0, "ap": 1.1, "t": "2026-10-23T20:00:00Z"}}
        )

    _broker(handler).get_quote("AAPL")
    assert seen[0].params["feed"] == "iex"
    assert "data.alpaca.markets" in str(seen[0])


def test_auth_headers_are_set():
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.headers)
        return httpx.Response(200, json={"cash": "1", "equity": "1", "buying_power": "1"})

    _broker(handler).get_account()
    assert captured["apca-api-key-id"] == "k"
    assert captured["apca-api-secret-key"] == "s"


def test_http_error_carries_status_and_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden: paper keys required")

    with pytest.raises(BrokerError) as excinfo:
        _broker(handler).get_account()
    assert "403" in str(excinfo.value)
    assert "paper keys required" in str(excinfo.value)


def test_empty_bars_payload_is_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"symbol": "XXXX", "bars": None, "next_page_token": None})

    assert _broker(handler).get_bars("XXXX") == []


def test_unknown_order_status_degrades_gracefully():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "o1",
                "symbol": "SPY",
                "qty": "1",
                "side": "buy",
                "type": "market",
                "status": "held",
                "filled_qty": "0",
                "filled_avg_price": None,
                "submitted_at": "2026-10-23T20:00:00Z",
            },
        )

    assert _broker(handler).get_order("o1").status is OrderStatus.UNKNOWN
