"""Contract tests: every broker must satisfy the same surface (decision D5).

The Alpaca implementation is tested against a fake HTTP transport, so the suite
runnable offline really does exercise the parsing, the order payload and the
error path — not just the mock broker.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tradingbench.brokers.alpaca_paper import AlpacaPaperBroker
from tradingbench.brokers.base import BrokerError
from tradingbench.brokers.mock import MockBroker
from tradingbench.models import OrderStatus, OrderType, Side


class FakeAlpaca:
    """Minimal stateful Alpaca stand-in served through httpx.MockTransport."""

    def __init__(self) -> None:
        self.orders: dict[str, dict] = {}
        self.requests: list[httpx.Request] = []
        self.cash = 10_000.0
        self.reject_orders = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if path == "/v2/account":
            return httpx.Response(
                200,
                json={
                    "id": "acct-123",
                    "account_number": "PA3TEST",
                    "cash": str(self.cash),
                    "equity": "10000",
                    "buying_power": "20000",
                    "currency": "USD",
                },
            )
        if path == "/v2/clock":
            return httpx.Response(200, json={"is_open": True, "next_open": "2026-10-26T13:30:00Z"})
        if path == "/v2/positions":
            return httpx.Response(
                200,
                json=[
                    {
                        "symbol": "AAPL",
                        "qty": "3",
                        "avg_entry_price": "180.00",
                        "current_price": "185.50",
                    }
                ],
            )
        if path.startswith("/v2/stocks/") and path.endswith("/quotes/latest"):
            return httpx.Response(
                200,
                json={
                    "symbol": "SPY",
                    "quote": {"bp": 654.10, "ap": 654.20, "t": "2026-10-23T19:59:58.123456789Z"},
                },
            )
        if path.startswith("/v2/stocks/") and path.endswith("/bars"):
            return httpx.Response(
                200,
                json={
                    "symbol": "SPY",
                    "bars": [
                        {
                            "t": "2026-10-23T04:00:00Z",
                            "o": 655.0,
                            "h": 657.0,
                            "l": 652.0,
                            "c": 654.1,
                            "v": 75000000,
                        },
                        {
                            "t": "2026-10-22T04:00:00Z",
                            "o": 650.0,
                            "h": 656.0,
                            "l": 648.5,
                            "c": 655.0,
                            "v": 81000000,
                        },
                    ],
                    "next_page_token": None,
                },
            )
        if path == "/v2/orders" and request.method == "POST":
            if self.reject_orders:
                return httpx.Response(422, json={"message": "insufficient buying power"})
            body = json.loads(request.content)
            order_id = f"order-{len(self.orders) + 1}"
            order = {
                "id": order_id,
                "symbol": body["symbol"],
                "qty": body["qty"],
                "side": body["side"],
                "type": body["type"],
                "status": "accepted",
                "filled_qty": "0",
                "filled_avg_price": None,
                "submitted_at": "2026-10-23T20:00:00Z",
            }
            self.orders[order_id] = order
            return httpx.Response(200, json=order)

        if path.startswith("/v2/orders/"):
            order_id = path.rsplit("/", 1)[-1]
            order = self.orders.get(order_id)
            if order is None:
                return httpx.Response(404, json={"message": "order not found"})
            if request.method == "DELETE":
                order["status"] = "canceled"
                return httpx.Response(204)
            return httpx.Response(200, json=order)

        return httpx.Response(404, json={"message": f"unmapped path {path}"})


def _alpaca(fake: FakeAlpaca) -> AlpacaPaperBroker:
    return AlpacaPaperBroker(
        api_key="test-key",
        secret_key="test-secret",
        transport=httpx.MockTransport(fake),
    )


@pytest.fixture(params=["mock", "alpaca-fake"])
def broker(request):
    if request.param == "mock":
        return MockBroker(symbols=("SPY", "AAPL", "NVDA"))
    return _alpaca(FakeAlpaca())


def test_account_is_coherent(broker):
    account = broker.get_account()
    assert account.cash > 0
    assert account.equity > 0
    assert account.currency == "USD"


def test_quote_has_positive_spread(broker):
    quote = broker.get_quote("SPY")
    assert quote.bid > 0
    assert quote.ask > quote.bid
    assert 0 <= quote.spread_bps < 500
    assert quote.timestamp is not None


def test_bars_are_chronological(broker):
    bars = broker.get_bars("SPY", limit=5)
    assert len(bars) >= 2
    assert bars == sorted(bars, key=lambda b: b.timestamp)
    assert all(bar.high >= max(bar.open, bar.close) for bar in bars)


def test_positions_are_mapped(broker):
    positions = broker.get_positions()
    assert isinstance(positions, list)
    for position in positions:
        assert position.qty > 0
        assert position.avg_entry_price > 0
        assert position.market_value == pytest.approx(position.qty * position.last_price)
    if isinstance(broker, MockBroker):
        assert positions == []  # a fresh mock account is flat


def test_market_order_roundtrip(broker):
    order = broker.place_order("SPY", 2, Side.BUY)
    assert order.side is Side.BUY
    assert order.order_type is OrderType.MARKET
    assert order.qty == 2
    assert broker.get_order(order.id).id == order.id


def test_orders_are_rejected_when_backend_says_no():
    fake = FakeAlpaca()
    fake.reject_orders = True
    broker = _alpaca(fake)
    with pytest.raises(BrokerError, match="422"):
        broker.place_order("SPY", 1, Side.BUY)


def test_cancel_is_idempotent_on_terminal_order(broker):
    if isinstance(broker, MockBroker):
        order = broker.place_order("SPY", 1, Side.BUY)
        broker.cancel_order(order.id)
        broker.cancel_order(order.id)  # must not raise
        assert broker.get_order(order.id).status is OrderStatus.FILLED
    else:
        order = broker.place_order("SPY", 1, Side.BUY)
        broker.cancel_order(order.id)
        assert broker.get_order(order.id).status is OrderStatus.CANCELED


def test_limit_order_carries_its_price():
    fake = FakeAlpaca()
    broker = _alpaca(fake)
    broker.place_order("SPY", 1, Side.BUY, order_type=OrderType.LIMIT, limit_price=600.0)
    posted = json.loads(fake.requests[-1].content)
    assert posted["type"] == "limit"
    assert posted["limit_price"] == "600.0"
    assert posted["time_in_force"] == "day"


def test_unknown_order_raises():
    broker = _alpaca(FakeAlpaca())
    with pytest.raises(BrokerError):
        broker.get_order("does-not-exist")


def test_missing_credentials_fail_loudly():
    with pytest.raises(BrokerError, match="credentials"):
        AlpacaPaperBroker(api_key="", secret_key="")
