"""Mapping tests: Alpaca JSON -> domain objects, and the failure path."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from tradingbench.brokers.alpaca_paper import AlpacaPaperBroker, _lookback_start
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


# -- barres : régression du bug trouvé sur le compte paper réel -----------------

def test_bars_request_sends_start_and_desc():
    """Sans `start`, Alpaca répond 200 avec bars:null. Avec asc+start, il renvoie
    les plus ANCIENNES barres. On vérifie donc les deux paramètres."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"bars": [], "symbol": "SPY"})

    _broker(handler).get_bars("SPY", limit=30)
    assert seen[0].params["sort"] == "desc"
    assert seen[0].params["start"]
    assert seen[0].params["limit"] == "30"


def test_bars_are_reversed_back_to_chronological_order():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "bars": [
                    {"t": "2026-10-23T04:00:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 1},
                    {"t": "2026-10-22T04:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1},
                ]
            },
        )

    bars = _broker(handler).get_bars("SPY")
    assert [bar.close for bar in bars] == [1.0, 2.0]
    assert bars[-1].timestamp.date().isoformat() == "2026-10-23"


def test_lookback_window_widens_with_the_requested_limit():
    now = datetime(2026, 10, 25, tzinfo=UTC)
    small = _lookback_start("1Day", 10, now)
    large = _lookback_start("1Day", 60, now)
    assert large < small < now
    assert (now - small).days > 10  # au moins 2 barres par jour demandé
    # 1Hour doit produire une fenêtre en heures, pas en dizaines de jours
    assert (now - _lookback_start("1Hour", 24, now)).days < 10
