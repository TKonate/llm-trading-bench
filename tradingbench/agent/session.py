"""One decision session: context, tools, orders, journal, critique.

Invariants held here:
- a session already run does not run again (key = date + slot);
- every order carries a deterministic `client_order_id`, so a double submit is
  impossible even if the process dies between "order sent" and "state saved";
- the day's token budget is checked before the call, never after;
- three consecutive tool failures pause the agent;
- everything is journaled as JSONL, including the decision to do nothing.

Deliberately not here: order sizing and risk limits (the executor enforces the
capital cap), and any knowledge of Alpaca (that stays behind the Broker).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
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

# UTC hour windows for the two daily sessions. They assume US summer time
# (EDT), where the session runs 13:30-20:00 UTC; in winter the real session
# starts and ends an hour later, so a window can fire outside it. The clock
# check in run_session is the authority; these ranges only keep the two runs
# apart within a day.
OPEN_HOURS = range(13, 16)
CLOSE_HOURS = range(17, 21)
MAX_TOOL_FAILURES = 3
CRITIQUE_EVERY = 5


@dataclass(slots=True)
class SessionOutcome:
    """What one session did, whether it ran or not.

    `ran=False` with a `reason` is a first-class result: a skipped session is a
    measurement too, and the caller has to be able to tell "chose not to trade"
    from "never executed".
    """

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
    """Current session slot, or None outside the windows. Weekdays only.

    Returns `(key, label)`: the key is what `AgentState.slots_done` stores, so it
    must be stable under a re-run and unique per day and per slot. The label is
    for the prompt and is not used for deduplication.
    """
    now = now.astimezone(UTC)
    if now.weekday() >= 5:
        return None
    if now.hour in OPEN_HOURS:
        return f"{now.date().isoformat()}:open", "ouverture"
    if now.hour in CLOSE_HOURS:
        return f"{now.date().isoformat()}:close", "fin de séance"
    return None


def _pct_change(values: list[float], window: int) -> float | None:
    """Percent change over `window` bars, or None when the history is too short.

    None rather than 0.0: the caller omits the figure from the prompt instead of
    telling the agent that a series it has never seen moved by zero.
    """
    if len(values) <= window or values[-window - 1] == 0:
        return None
    return (values[-1] / values[-window - 1] - 1) * 100


# Above 2 % of the mid, a spread no longer describes a liquid equity quote: it
# is a stale or crossed book, and the session prints a warning instead of a
# price the agent might trade on. Observed on real data outside trading hours.
MAX_TRADABLE_SPREAD_BPS = 200.0


def _quote_line(symbol: str, quote, closes: list[float]) -> str:
    """Describe one quote, and refuse to present as tradable what is not.

    Seen on real off-hours data: an ask at 0.00 and spreads around 1 000 bps. An
    agent that reads that as a tradable price takes a destructive decision, so
    the line says "do not trade" instead of letting the model infer it.
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
    """One line per symbol: quote, weekly and monthly change, spread.

    25 bars are requested because the monthly change needs 21 closes. A symbol
    that fails is reported as unavailable and the loop continues: a single dead
    ticker must not cancel the session, and the agent has to see the gap rather
    than a shorter universe.
    """
    lines: list[str] = []
    for symbol in universe:
        try:
            quote = broker.get_quote(symbol)
            closes = [bar.close for bar in broker.get_bars(symbol, limit=25)]
        except Exception as exc:  # noqa: BLE001 - one dead symbol must not kill it
            lines.append(f"- {symbol} : donnée indisponible ({exc})")
            continue
        lines.append(_quote_line(symbol, quote, closes))
    return "\n".join(lines) if lines else "(aucune donnée de marché disponible)"


def build_portfolio_context(broker: Broker, budget_usd: float | None = None) -> str:
    """Cash, equity, exposure against the reference budget, and open positions.

    `budget_usd` is optional because the same function serves the prompt (where
    the agent must see how much of its budget is used) and the self-critique
    (where only equity and P&L matter). Exposure over the budget is shown, not
    clamped: the executor is what refuses the order, and the agent has to see
    the real state.
    """
    try:
        account = broker.get_account()
        positions = broker.get_positions()
    except Exception as exc:  # noqa: BLE001 - same rule: degrade, never abort
        return f"(portefeuille indisponible : {exc})"
    lines = [f"cash {account.cash:,.2f} {account.currency} · equity {account.equity:,.2f}"]
    if budget_usd:
        exposure = sum(position.market_value for position in positions)
        share = 100 * exposure / budget_usd if budget_usd else 0.0
        lines.append(
            f"exposition {exposure:,.2f} $ sur un budget de {budget_usd:,.2f} $ ({share:.0f} %)"
        )
    if not positions:
        lines.append("aucune position ouverte")
    for position in positions:
        lines.append(
            f"- {position.symbol} : {position.qty:g} @ {position.avg_entry_price:.2f} "
            f"(valeur {position.market_value:,.2f}, P&L {position.unrealized_pl:+,.2f})"
        )
    return "\n".join(lines)


def build_history(state: AgentState) -> str:
    """The last five decisions, so the agent can stay consistent with its own plan.

    Five is one critique window (CRITIQUE_EVERY): the trades the next critique
    will comment on are the ones shown here.
    """
    if not state.recent_trades:
        return "(aucun ordre passé pour l'instant)"
    lines = []
    for trade in state.recent_trades[-5:]:
        lines.append(
            f"- {trade.get('ts', '?')} {trade.get('side', '?')} {trade.get('qty', '?')} "
            f"{trade.get('symbol', '?')} — {trade.get('rationale', 'sans justification')}"
        )
    return "\n".join(lines)


def format_system_prompt(
    strategy: str = DEFAULT_STRATEGY, capital_eur: float = 1000.0, eur_usd: float = 1.08
) -> str:
    """Fill the system prompt with the capital figures the agent must respect.

    0.25 is the position cap the prose states as "25 % du capital": the two have
    to be edited together, or the agent is told one limit and measured against
    another. Thousands are separated by a space because the prompt is French.
    """
    capital_usd = capital_eur * eur_usd
    max_position_eur = capital_eur * 0.25
    return SYSTEM_PROMPT.format(
        universe=", ".join(UNIVERSE),
        benchmark=BENCHMARK,
        strategy=strategy,
        max_positions=MAX_POSITIONS,
        capital=f"{capital_eur:,.0f} € (≈ {capital_usd:,.0f} $, comptes en dollars)".replace(
            ",", " "
        ),
        max_position=f"{max_position_eur:,.0f} €".replace(",", " "),
    )


def build_news_context(broker: Broker, limit: int = 4) -> str:
    """Recent headlines, marked untrusted in the template that receives them.

    A wire story can contain text that imitates an instruction; the label is
    what tells the agent to read it as data and to ignore it as an order.
    """
    try:
        items = broker.get_news(list(UNIVERSE), limit=limit)
    except Exception as exc:  # noqa: BLE001 - missing news must not block the session
        return f"(actualités indisponibles : {exc})"
    if not items:
        return "(aucune actualité récente sur l'univers)"
    return "\n".join(f"- {item.one_line()}" for item in items[:limit])


def read_veille(path: Path | None, now: datetime | None = None, max_age_days: int = 4) -> str:
    """Third-party intelligence note, from a Markdown file; the most recent wins.

    With a directory, files whose name starts with `_` are skipped and the
    newest by mtime is used. The age is printed because a three-week-old note is
    not the same input as yesterday's, and the agent has to be able to judge it.
    Content is capped at 3 500 characters: this block competes with the portfolio
    for the context window.
    """
    if path is None:
        return "(aucune veille configurée — VEILLE_PATH absent du .env)"
    candidate = Path(path)
    if candidate.is_dir():
        files = sorted(
            (f for f in candidate.glob("*.md") if not f.name.startswith("_")),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return f"(dossier de veille vide : {candidate})"
        candidate = files[0]
    if not candidate.exists():
        return f"(pas encore de veille : {candidate} n'existe pas)"
    age_days = ((now or datetime.now(UTC)).timestamp() - candidate.stat().st_mtime) / 86400
    # `errors="replace"`: a hand-edited file can carry a stray byte, and a
    # decoding error here would cost the whole session.
    content = candidate.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return f"(veille vide : {candidate.name})"
    header = f"Source : {candidate.name}, âge {age_days:.1f} jour(s)."
    if age_days > max_age_days:
        header += " ⚠ VEILLE PÉRIMÉE, à ne pas traiter comme une information de marché actuelle."
    return f"{header}\n{content[:3500]}"


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
    capital_eur: float = 1000.0,
    eur_usd: float = 1.08,
    veille_path: Path | None = None,
) -> SessionOutcome:
    """Run one decision session and return what happened.

    Gates run in this order: slot, slot already run, paused agent, daily token
    budget, market clock. The order is not arbitrary: an exhausted budget or a
    paused agent is still reported when the market is closed too, and the budget
    is checked before the first LLM call because one checked afterwards has
    already been spent.

    `force=True` bypasses the clock and the window, never the deduplication: a
    forced run records its slot key, so forcing twice in one day returns a skip
    instead of placing a second set of orders.

    `state` is mutated in place (tokens, slots, trades) and persisting it is the
    caller's job, which keeps the gap between "order placed" and "state saved"
    down to one call.
    """
    now = now or datetime.now(UTC)
    state.roll_day(now.date())
    agent = state.agent_id

    slot_info = session_slot(now)
    if slot_info is None and not force:
        outcome = SessionOutcome(
            agent, slot="hors-fenêtre", ran=False, reason="hors créneau ou jour non ouvré"
        )
        # Log the refusals too: without this line, "decided nothing" and "never
        # ran" are indistinguishable in the logs.
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

    # The budget is a constraint on the measurement, not decoration: a paper
    # account left at 100 000 $ would make any performance percentage absurd.
    capital_usd = round(capital_eur * eur_usd, 2)
    try:
        equity = broker.get_account().equity
    except Exception as exc:  # noqa: BLE001
        outcome = SessionOutcome(agent, slot=slot, ran=False, reason=f"compte illisible : {exc}")
        logger.log("session_skipped", agent=agent, slot=slot, reason=outcome.reason)
        return outcome
    if state.starting_equity_usd is None:
        state.starting_equity_usd = equity
    if equity > capital_usd * 1.2:
        # Not blocking: the executor's exposure cap is what enforces the 1 000 €
        # budget. The trace is kept because a funded paper account flatters every
        # percentage that is read from it later.
        logger.log(
            "capital_warning",
            agent=agent,
            slot=slot,
            account_equity=equity,
            budget_usd=capital_usd,
            note="compte paper au-dessus du budget : exposition plafonnée par le code",
        )

    # The slot is part of the order prefix: the same agent retrying a slot gets
    # the same ids, and a second slot cannot collide with the first.
    executor = ToolExecutor(
        broker, client_order_prefix=f"{agent}-{slot}", max_notional=capital_usd
    )
    memory = f"TON PLAN EN COURS (auto-critique)\n{state.plan}\n\n" if state.plan else ""
    user_message = CONTEXT_TEMPLATE.format(
        slot=f"{slot} ({label})",
        now=now.strftime("%Y-%m-%d %H:%M"),
        market=build_market_context(broker),
        news=build_news_context(broker),
        veille=read_veille(veille_path, now),
        portfolio=build_portfolio_context(broker, capital_usd),
        history=build_history(state),
        memory=memory,
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": format_system_prompt(strategy, capital_eur, eur_usd),
        },
        {"role": "user", "content": user_message},
    ]

    tool_calls: list[str] = []
    tool_failures = 0
    tokens = 0
    text = ""
    completion: Completion | None = None

    for _ in range(max_turns):
        # One turn per round: an agent that reads the price then decides needs at
        # least two, and `max_turns` is what bounds the cost of a model that
        # loops instead of answering.
        completion = client.complete(messages, tools=TOOLS, max_tokens=1200)
        tokens += completion.total_tokens
        # Keep the last non-empty text: the turn that calls a tool usually
        # carries no prose, and it would otherwise erase the reasoning of the
        # turn before it.
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
                        # `raw_arguments`, not the parsed dict: the transcript has
                        # to replay the exact string the model produced, or a
                        # strict provider rejects the history on the next turn.
                        "function": {"name": call.name, "arguments": call.raw_arguments},
                    }
                    for call in completion.tool_calls
                ],
            }
        )
        for call in completion.tool_calls:
            tool_calls.append(call.name)
            payload = executor.dispatch(call.name, call.arguments)
            # The payload is already a JSON string, and dispatch encodes every
            # failure as {"error": ...}; a substring test avoids parsing it back
            # just to count failures.
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
            # One rationale per session, attached to every order it placed: the
            # model justifies the decision, not each leg of it. Truncated to 400
            # characters so a rambling answer cannot bloat the state and journal.
            "rationale": (text or "").strip()[:400],
        }
        orders.append(record)
        state.record_trade(record)

    state.record_call(tokens, tool_failures)
    # Marked done only now: a crash during the turn loop leaves the slot unmarked
    # so the session can be retried, and the client_order_id makes that retry
    # safe rather than duplicating the orders.
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
    """Ask the agent to review its last trades and store the resulting plan.

    The plan is capped at 1 500 characters because it is re-injected at the top
    of every following prompt: an unbounded plan would slowly eat the context
    the market data needs.

    The critique is sent with the default system prompt, not with the session's
    capital figures: with a non-default capital, the amounts quoted here will not
    match the ones the trades were made under.
    """
    trades = "\n".join(
        f"- {item.get('ts')} {item.get('side')} {item.get('qty')} {item.get('symbol')} "
        f"(statut {item.get('status')}) — {item.get('rationale', '')}"
        for item in state.recent_trades[-CRITIQUE_EVERY:]
    )
    try:
        performance = build_portfolio_context(broker)
    except BrokerError:  # pragma: no cover - the builder already degrades internally
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
    """Readable JSON view of an order list, for post-mortem scripts."""
    return json.dumps(orders, ensure_ascii=False, indent=2)
