"""Broker factory — the only place that knows which backends exist."""

from __future__ import annotations

from ..config import Settings
from .alpaca_paper import AlpacaPaperBroker
from .base import Broker, BrokerError
from .mock import MockBroker

__all__ = ["AlpacaPaperBroker", "Broker", "BrokerError", "MockBroker", "get_broker"]


def get_broker(kind: str, settings: Settings | None = None) -> Broker:
    settings = settings or Settings.from_env()
    if kind == "mock":
        return MockBroker()
    if kind in {"alpaca", "alpaca-paper", "paper"}:
        if not settings.has_alpaca_credentials:
            raise BrokerError(
                "Alpaca credentials missing: create a free paper account on "
                "https://app.alpaca.markets then set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in .env"
            )
        return AlpacaPaperBroker(
            api_key=settings.alpaca_api_key or "",
            secret_key=settings.alpaca_secret_key or "",
            feed=settings.alpaca_feed,
        )
    raise BrokerError(f"unknown broker {kind!r} (expected 'mock' or 'alpaca')")
