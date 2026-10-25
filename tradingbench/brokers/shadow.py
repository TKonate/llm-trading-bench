"""Broker « fantôme » pour les répétitions à blanc.

Lecture → broker réel (prix et portefeuille vrais). Écriture → simulation locale
initialisée avec l'état réel. On peut ainsi dérouler une session complète avec
des données de marché authentiques sans toucher au compte paper, ni au futur compte
réel.
"""

from __future__ import annotations

from ..models import Account, Bar, NewsItem, Order, OrderType, Position, Quote, Side
from .base import Broker
from .mock import MockBroker


class ShadowBroker(Broker):
    name = "shadow"
    is_paper = True

    def __init__(self, real: Broker, symbols: tuple[str, ...]) -> None:
        self._real = real
        account = real.get_account()
        self._shadow = MockBroker(symbols=symbols, cash=account.cash)
        for position in real.get_positions():
            self._shadow.seed_position(position.symbol, position.qty, position.avg_entry_price)
        for symbol in symbols:
            try:
                self._shadow.set_price(symbol, real.get_quote(symbol).mid)
            except Exception:  # noqa: BLE001 - un symbole indisponible ne bloque pas la répétition
                continue

    # lectures réelles
    def get_quote(self, symbol: str) -> Quote:
        return self._real.get_quote(symbol)

    def get_bars(self, symbol: str, *, timeframe: str = "1Day", limit: int = 30) -> list[Bar]:
        return self._real.get_bars(symbol, timeframe=timeframe, limit=limit)

    def get_news(self, symbols: list[str], *, limit: int = 5) -> list[NewsItem]:
        return self._real.get_news(symbols, limit=limit)

    def is_market_open(self) -> bool:
        return self._real.is_market_open()

    # écritures simulées
    def get_account(self) -> Account:
        return self._shadow.get_account()

    def get_positions(self) -> list[Position]:
        return self._shadow.get_positions()

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
        return self._shadow.place_order(
            symbol,
            qty,
            side,
            order_type=order_type,
            limit_price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def get_order(self, order_id: str) -> Order:
        return self._shadow.get_order(order_id)

    def cancel_order(self, order_id: str) -> None:
        self._shadow.cancel_order(order_id)
