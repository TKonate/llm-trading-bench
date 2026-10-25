"""Exécution réelle des outils appelés par le modèle, sur un broker de test.

Un harnais mono-tour juge le modèle sur sa première réaction. En réalité, un
agent consulte le prix, lit son portefeuille, puis décide : refuser ce
enchaînement revient à mesurer la myopie du harnais, pas la compétence du modèle.

L'exécution se fait sur le MockBroker : aucun appel réseau, aucun argent, mais
les outils renvoient de vraies structures (prix, positions, erreurs de broker).
"""

from __future__ import annotations

import json
from typing import Any

from ..brokers.base import Broker, BrokerError
from ..brokers.mock import MockBroker
from ..models import OrderType, Side


class ToolExecutor:
    def __init__(
        self, broker: Broker | None = None, client_order_prefix: str | None = None
    ) -> None:
        self.broker = broker or MockBroker()
        self.client_order_prefix = client_order_prefix
        self.order_attempts: list[dict[str, Any]] = []
        self.order_results: list[dict[str, Any]] = []

    def seed(self, positions: tuple[tuple[str, float], ...]) -> None:
        """Ouvre des positions avant le scénario (vente, arbitrage de portefeuille)."""
        seed_position = getattr(self.broker, "seed_position", None)
        if seed_position is None:
            return
        for symbol, qty in positions:
            seed_position(symbol, qty)

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Exécute un appel d'outil et renvoie le résultat sérialisé pour le modèle."""
        try:
            result = self._call(name, arguments)
        except BrokerError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - une erreur d'outil n'arrête pas la boucle
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False, default=str)

    def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "get_quote":
            quote = self.broker.get_quote(str(arguments["symbol"]).upper())
            return {
                "symbol": quote.symbol,
                "bid": quote.bid,
                "ask": quote.ask,
                "mid": round(quote.mid, 4),
                "spread_bps": round(quote.spread_bps, 2),
            }
        if name == "get_bars":
            bars = self.broker.get_bars(
                str(arguments["symbol"]).upper(), limit=int(arguments.get("limit", 30))
            )
            return {
                "symbol": arguments["symbol"].upper(),
                "bars": [
                    {
                        "t": bar.timestamp.date().isoformat(),
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                    for bar in bars[-20:]
                ],
            }
        if name == "get_portfolio":
            account = self.broker.get_account()
            positions = self.broker.get_positions()
            return {
                "cash": account.cash,
                "equity": account.equity,
                "positions": [
                    {
                        "symbol": position.symbol,
                        "qty": position.qty,
                        "avg_entry_price": position.avg_entry_price,
                        "market_value": round(position.market_value, 2),
                    }
                    for position in positions
                ],
            }
        if name == "place_order":
            side = Side(str(arguments["side"]).lower())
            order_type = OrderType(str(arguments.get("order_type", "market")).lower())
            qty = float(arguments["qty"])
            client_order_id = None
            if self.client_order_prefix:
                # Idempotence : le même identifiant rejoué ne crée pas un second ordre.
                client_order_id = f"{self.client_order_prefix}-{len(self.order_attempts)}"
            self.order_attempts.append(
                {"symbol": arguments["symbol"], "qty": qty, "side": side.value}
            )
            order = self.broker.place_order(
                str(arguments["symbol"]).upper(),
                qty,
                side,
                order_type=order_type,
                limit_price=arguments.get("limit_price"),
                client_order_id=client_order_id,
            )
            result = {
                "id": order.id[:8],
                "status": order.status.value,
                "filled_avg_price": order.filled_avg_price,
            }
            self.order_results.append(
                {
                    **result,
                    "symbol": arguments["symbol"].upper(),
                    "qty": qty,
                    "side": side.value,
                }
            )
            return result
        return {"error": f"outil inconnu : {name}"}
