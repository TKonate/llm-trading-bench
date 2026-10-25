"""Une session de décision : contexte → outils → ordres → journal → critique.

Invariants tenus ici :
- une session déjà exécutée ne rejoue pas (clé = date + créneau) ;
- chaque ordre porte un `client_order_id` déterministe (double envoi impossible) ;
- le budget de tokens du jour est vérifié *avant* l'appel, pas après ;
- trois échecs d'outils consécutifs mettent l'agent en pause ;
- tout est journalisé en JSONL, y compris les décisions de ne rien faire.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..brokers.base import Broker, BrokerError
from ..harness.executor import ToolExecutor
from ..harness.scenarios import TOOLS
from ..jsonlog import JsonlLogger
from ..llm.client import ChatClient, Completion
from .prompts import (
    BENCHMARK,
    CONTEXT_TEMPLATE,
    CRITIQUE_PROMPT,
    DEFAULT_STRATEGY,
    MAX_POSITIONS,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    UNIVERSE,
)
from .state import AgentState

OPEN_HOURS = range(13, 16)  # 13:30–20:00 UTC en heure d'été US ; on vise l'ouverture
CLOSE_HOURS = range(17, 21)
MAX_TOOL_FAILURES = 3
CRITIQUE_EVERY = 5


@dataclass(slots=True)
class SessionOutcome:
    agent: str
    slot: str
    ran: bool
    reason: str = ""
    orders: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    text: str = ""
    critique: str | None = None
    tool_calls: list[str] = field(default_factory=list)


def session_slot(now: datetime) -> tuple[str, str] | None:
    """Créneau de la session courante, ou None hors fenêtre. Jours ouvrables uniquement."""
    now = now.astimezone(UTC)
    if now.weekday() >= 5:
        return None
    if now.hour in OPEN_HOURS:
        return f"{now.date().isoformat()}:open", "ouverture"
    if now.hour in CLOSE_HOURS:
        return f"{now.date().isoformat()}:close", "fin de séance"
    return None


def _pct_change(values: list[float], window: int) -> float | None:
    if len(values) <= window or values[-window - 1] == 0:
        return None
    return (values[-1] / values[-window - 1] - 1) * 100


MAX_TRADABLE_SPREAD_BPS = 200.0


def _quote_line(symbol: str, quote, closes: list[float]) -> str:
    """Décrit une cotation — et refuse de présenter comme tradable ce qui ne l'est pas.

    Constaté sur de vraies données hors séance : ask à 0,00 et spreads à 1 000 bps.
    Un agent qui lit ça comme un prix tradable prend une décision destructrice.
    """
    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        return (
            f"- {symbol} : COTATION INVALIDE "
            f"(bid {quote.bid:.2f} / ask {quote.ask:.2f}) — ne pas trader"
        )
    parts = [f"- {symbol} : bid {quote.bid:.2f} / ask {quote.ask:.2f}"]
    week = _pct_change(closes, 5)
    month = _pct_change(closes, 20)
    if week is not None:
        parts.append(f"5 j {week:+.1f}%")
    if month is not None:
        parts.append(f"20 j {month:+.1f}%")
    parts.append(f"spread {quote.spread_bps:.1f} bps")
    if quote.spread_bps > MAX_TRADABLE_SPREAD_BPS:
        parts.append("⚠ spread anormal, cotation probablement périmée — ne pas trader")
    return " · ".join(parts)


def build_market_context(broker: Broker, universe: tuple[str, ...] = UNIVERSE) -> str:
    lines: list[str] = []
    for symbol in universe:
        try:
            quote = broker.get_quote(symbol)
            closes = [bar.close for bar in broker.get_bars(symbol, limit=25)]
        except Exception as exc:  # noqa: BLE001 - un symbole muet ne doit pas annuler la session
            lines.append(f"- {symbol} : donnée indisponible ({exc})")
            continue
        lines.append(_quote_line(symbol, quote, closes))
    return "\n".join(lines) if lines else "(aucune donnée de marché disponible)"


def build_portfolio_context(broker: Broker) -> str:
    try:
        account = broker.get_account()
        positions = broker.get_positions()
    except Exception as exc:  # noqa: BLE001 - idem : on dégrade, on n'interrompt pas
        return f"(portefeuille indisponible : {exc})"
    lines = [f"cash {account.cash:,.2f} {account.currency} · equity {account.equity:,.2f}"]
    if not positions:
        lines.append("aucune position ouverte")
    for position in positions:
        lines.append(
            f"- {position.symbol} : {position.qty:g} @ {position.avg_entry_price:.2f} "
            f"(valeur {position.market_value:,.2f}, P&L {position.unrealized_pl:+,.2f})"
        )
    return "\n".join(lines)


def build_history(state: AgentState) -> str:
    if not state.recent_trades:
        return "(aucun ordre passé pour l'instant)"
    lines = []
    for trade in state.recent_trades[-5:]:
        lines.append(
            f"- {trade.get('ts', '?')} {trade.get('side', '?')} {trade.get('qty', '?')} "
            f"{trade.get('symbol', '?')} — {trade.get('rationale', 'sans justification')}"
        )
    return "\n".join(lines)


def format_system_prompt(strategy: str = DEFAULT_STRATEGY) -> str:
    return SYSTEM_PROMPT.format(
        universe=", ".join(UNIVERSE),
        benchmark=BENCHMARK,
        strategy=strategy,
        max_positions=MAX_POSITIONS,
    )


def run_session(
    broker: Broker,
    client: ChatClient,
    state: AgentState,
    logger: JsonlLogger,
    *,
    now: datetime | None = None,
    force: bool = False,
    budget_tokens: int = 120_000,
    max_turns: int = 6,
    strategy: str = DEFAULT_STRATEGY,
) -> SessionOutcome:
    now = now or datetime.now(UTC)
    state.roll_day(now.date())
    agent = state.agent_id

    slot_info = session_slot(now)
    if slot_info is None and not force:
        outcome = SessionOutcome(
            agent, slot="hors-fenêtre", ran=False, reason="hors créneau ou jour non ouvré"
        )
        # Journaliser aussi les refus : sans cette ligne, « n'a rien fait » et
        # « n'a pas tourné » sont indistinguables dans les logs.
        logger.log("session_skipped", agent=agent, slot=outcome.slot, reason=outcome.reason)
        return outcome
    slot, label = slot_info if slot_info else (f"{now.date().isoformat()}:force", "forcée")

    if state.is_slot_done(slot):
        outcome = SessionOutcome(agent, slot=slot, ran=False, reason="session déjà exécutée")
        logger.log("session_skipped", agent=agent, slot=slot, reason=outcome.reason)
        return outcome

    if state.consecutive_tool_failures >= MAX_TOOL_FAILURES:
        outcome = SessionOutcome(
            agent, slot=slot, ran=False, reason="pause auto : échecs d'outils consécutifs"
        )
        logger.log("session_skipped", agent=agent, slot=slot, reason=outcome.reason)
        return outcome

    if state.tokens_spent_today >= budget_tokens:
        outcome = SessionOutcome(
            agent, slot=slot, ran=False, reason="budget de tokens du jour atteint"
        )
        logger.log("session_skipped", agent=agent, slot=slot, reason=outcome.reason)
        return outcome

    if not force and not broker.is_market_open():
        outcome = SessionOutcome(agent, slot=slot, ran=False, reason="marché fermé")
        logger.log("session_skipped", agent=agent, slot=slot, reason=outcome.reason)
        return outcome

    executor = ToolExecutor(broker, client_order_prefix=f"{agent}-{slot}")
    memory = f"TON PLAN EN COURS (auto-critique)\n{state.plan}\n\n" if state.plan else ""
    user_message = CONTEXT_TEMPLATE.format(
        slot=f"{slot} ({label})",
        now=now.strftime("%Y-%m-%d %H:%M"),
        market=build_market_context(broker),
        portfolio=build_portfolio_context(broker),
        history=build_history(state),
        memory=memory,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": format_system_prompt(strategy)},
        {"role": "user", "content": user_message},
    ]

    tool_calls: list[str] = []
    tool_failures = 0
    tokens = 0
    text = ""
    completion: Completion | None = None

    for _ in range(max_turns):
        completion = client.complete(messages, tools=TOOLS, max_tokens=1200)
        tokens += completion.total_tokens
        text = completion.text or text
        if not completion.tool_calls:
            break
        messages.append(
            {
                "role": "assistant",
                "content": completion.text or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.raw_arguments},
                    }
                    for call in completion.tool_calls
                ],
            }
        )
        for call in completion.tool_calls:
            tool_calls.append(call.name)
            payload = executor.dispatch(call.name, call.arguments)
            if '"error"' in payload:
                tool_failures += 1
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": payload}
            )

    orders = []
    for order in executor.order_results:
        record = {
            "ts": now.isoformat(timespec="seconds"),
            "symbol": order["symbol"],
            "qty": order["qty"],
            "side": order["side"],
            "status": order["status"],
            "filled_avg_price": order["filled_avg_price"],
            "rationale": (text or "").strip()[:400],
        }
        orders.append(record)
        state.record_trade(record)

    state.record_call(tokens, tool_failures)
    state.mark_slot_done(slot)
    state.prompt_version = PROMPT_VERSION

    critique = None
    if state.trades_since_critique >= CRITIQUE_EVERY:
        critique = _self_critique(client, state, broker, logger)
        state.trades_since_critique = 0

    logger.log(
        "session",
        agent=agent,
        slot=slot,
        tokens=tokens,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        orders=orders,
        text=(text or "")[:800],
        critique=critique,
    )

    return SessionOutcome(
        agent=agent,
        slot=slot,
        ran=True,
        reason="ok",
        orders=orders,
        tokens=tokens,
        text=(text or "").strip()[:800],
        critique=critique,
        tool_calls=tool_calls,
    )


def _self_critique(
    client: ChatClient, state: AgentState, broker: Broker, logger: JsonlLogger
) -> str:
    trades = "\n".join(
        f"- {item.get('ts')} {item.get('side')} {item.get('qty')} {item.get('symbol')} "
        f"(statut {item.get('status')}) — {item.get('rationale', '')}"
        for item in state.recent_trades[-CRITIQUE_EVERY:]
    )
    try:
        performance = build_portfolio_context(broker)
    except BrokerError:  # pragma: no cover - défensif
        performance = "(indisponible)"
    prompt = CRITIQUE_PROMPT.format(
        count=min(len(state.recent_trades), CRITIQUE_EVERY),
        trades=trades,
        performance=performance,
    )
    completion = client.complete(
        [
            {"role": "system", "content": format_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
    )
    plan = (completion.text or "").strip()
    state.plan = plan[:1500]
    state.critique_count += 1
    logger.log("critique", agent=state.agent_id, plan=state.plan, tokens=completion.total_tokens)
    return plan


def orders_as_json(orders: list[dict[str, Any]]) -> str:
    return json.dumps(orders, ensure_ascii=False, indent=2)
