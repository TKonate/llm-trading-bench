"""Session loop: scheduling, idempotency, budget, auto-pause, self-critique."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pytest

from tradingbench.agent.prompts import DEFAULT_STRATEGY, PROMPT_VERSION
from tradingbench.agent.session import (
    build_market_context,
    build_news_context,
    build_portfolio_context,
    format_system_prompt,
    run_session,
    session_slot,
)
from tradingbench.agent.state import AgentState, load_state, save_state
from tradingbench.brokers.mock import MockBroker
from tradingbench.jsonlog import JsonlLogger
from tradingbench.llm.client import Completion, ToolCall
from tradingbench.models import Side

MONDAY_OPEN = datetime(2026, 10, 26, 14, 0, tzinfo=UTC)  # lundi, 14h UTC
MONDAY_CLOSE = datetime(2026, 10, 26, 18, 30, tzinfo=UTC)
SUNDAY = datetime(2026, 10, 25, 14, 0, tzinfo=UTC)


def trade_call(symbol="SPY", qty=1, side="buy") -> ToolCall:
    args = {"symbol": symbol, "qty": qty, "side": side}
    return ToolCall(id="call-1", name="place_order", arguments=args, raw_arguments=json.dumps(args))


class BuyThenStop:
    """Un modèle minimal : un ordre, puis une justification."""

    model = "scripted"

    def __init__(self, trade: ToolCall | None = None) -> None:
        self.trade = trade or trade_call()
        self.calls = 0

    def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=800):
        self.calls += 1
        if self.calls == 1 and self.trade is not None:
            return Completion(
                text="",
                tool_calls=[self.trade],
                model=self.model,
                finish_reason="tool_calls",
                prompt_tokens=60, completion_tokens=40,
            )
        return Completion(
            text="J'achète SPY car la tendance 20 jours est haussière.",
            tool_calls=[],
            model=self.model,
            finish_reason="stop",
            prompt_tokens=30, completion_tokens=20,
        )


class CritiqueClient:
    """Répond aux appels de session puis à l'auto-critique."""

    model = "scripted"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=800):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "auto-critique" in prompt:
            return Completion(
                text=(
                    "1. Bien : discipline. 2. Mal : trop concentré. "
                    "3. Plan : réduire à 2 positions."
                ),
                tool_calls=[],
                model=self.model,
                finish_reason="stop",
                prompt_tokens=50, completion_tokens=30,
            )
        return Completion(
            text="", tool_calls=[], model=self.model, finish_reason="stop",
            prompt_tokens=12, completion_tokens=8,
        )


# -- créneaux -------------------------------------------------------------------

def test_slot_is_none_on_weekends():
    assert session_slot(SUNDAY) is None


def test_open_and_close_slots_are_distinct():
    assert session_slot(MONDAY_OPEN) == ("2026-10-26:open", "ouverture")
    assert session_slot(MONDAY_CLOSE) == ("2026-10-26:close", "fin de séance")


def test_slot_is_none_outside_market_hours():
    assert session_slot(datetime(2026, 10, 26, 9, 0, tzinfo=UTC)) is None


# -- contexte -------------------------------------------------------------------

def test_market_context_contains_every_symbol_and_no_invention():
    context = build_market_context(MockBroker(cash=1080.0))
    for symbol in ("AAPL", "MSFT", "NVDA", "SPY"):
        assert symbol in context
    assert "bid" in context and "spread" in context


def test_market_context_survives_a_failing_symbol():
    class Broken(MockBroker):
        def get_quote(self, symbol):
            raise RuntimeError("data down")

    assert "indisponible" in build_market_context(Broken())


def test_portfolio_context_of_a_flat_account():
    assert "aucune position" in build_portfolio_context(MockBroker(cash=1080.0))


class BadQuotes(MockBroker):
    """Reproduit ce que la vraie API a renvoyé hors séance."""

    def get_quote(self, symbol):
        from datetime import UTC, datetime

        from tradingbench.models import Quote

        if symbol == "NVDA":
            return Quote(symbol="NVDA", bid=0.0, ask=0.0, timestamp=datetime.now(UTC))
        if symbol == "AAPL":
            return Quote(symbol="AAPL", bid=200.0, ask=220.0, timestamp=datetime.now(UTC))
        return super().get_quote(symbol)


def test_a_zero_quote_is_labelled_as_untradeable():
    context = build_market_context(BadQuotes())
    assert "COTATION INVALIDE" in context


def test_an_absurd_spread_is_flagged():
    context = build_market_context(BadQuotes())
    assert "spread anormal" in context


def test_system_prompt_pins_the_rules():
    prompt = format_system_prompt()
    assert "TSLA" not in prompt
    assert "SPY" in prompt and "maximum 3 positions" in prompt.lower()
    assert DEFAULT_STRATEGY.split(".")[0] in prompt


# -- session --------------------------------------------------------------------

def test_session_places_the_order_and_records_it(tmp_path):
    state = AgentState(agent_id="test")
    logger = JsonlLogger(tmp_path / "agent.jsonl")
    outcome = run_session(
        MockBroker(cash=1080.0), BuyThenStop(), state, logger, now=MONDAY_OPEN, force=True
    )
    assert outcome.ran
    assert len(outcome.orders) == 1
    assert outcome.orders[0]["symbol"] == "SPY"
    assert state.total_trades == 1
    assert state.is_slot_done("2026-10-26:open")
    assert state.prompt_version == PROMPT_VERSION
    events = [json.loads(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()]
    assert any(event["event"] == "session" for event in events)


def test_the_same_slot_never_runs_twice(tmp_path):
    state = AgentState(agent_id="test")
    logger = JsonlLogger(tmp_path / "agent.jsonl")
    first = run_session(
        MockBroker(cash=1080.0), BuyThenStop(), state, logger, now=MONDAY_OPEN, force=True
    )
    second = run_session(
        MockBroker(cash=1080.0), BuyThenStop(), state, logger, now=MONDAY_OPEN, force=True
    )
    assert first.ran
    assert not second.ran
    assert "déjà exécutée" in second.reason
    assert state.total_trades == 1  # aucun second ordre


def test_session_is_skipped_when_the_market_is_closed(tmp_path):
    broker = MockBroker(cash=1080.0, market_open=False)
    outcome = run_session(
        broker, BuyThenStop(), AgentState(agent_id="test"), JsonlLogger(tmp_path / "a.jsonl"),
        now=MONDAY_OPEN,
    )
    assert not outcome.ran
    assert "marché fermé" in outcome.reason


def test_session_is_skipped_outside_the_window(tmp_path):
    log_path = tmp_path / "a.jsonl"
    outcome = run_session(
        MockBroker(cash=1080.0),
        BuyThenStop(),
        AgentState(agent_id="test"),
        JsonlLogger(log_path),
        now=SUNDAY,
    )
    assert not outcome.ran
    assert "créneau" in outcome.reason
    # Un refus doit laisser une trace : sinon on ne sait pas si le job a tourné.
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert events and events[-1]["event"] == "session_skipped"
    assert "créneau" in events[-1]["reason"]


def test_every_skip_path_is_journalised(tmp_path):
    """Marché fermé, budget atteint, pause auto : chacun doit écrire une ligne."""
    log_path = tmp_path / "skip.jsonl"
    logger = JsonlLogger(log_path)

    run_session(
        MockBroker(cash=1080.0, market_open=False),
        BuyThenStop(),
        AgentState(agent_id="t1"),
        logger,
        now=MONDAY_OPEN,
    )
    run_session(
        MockBroker(cash=1080.0),
        BuyThenStop(),
        AgentState(agent_id="t2", tokens_spent_today=999_999, budget_day="2026-10-26"),
        logger,
        now=MONDAY_OPEN,
        force=True,
    )
    run_session(
        MockBroker(cash=1080.0),
        BuyThenStop(),
        AgentState(agent_id="t3", consecutive_tool_failures=3),
        logger,
        now=MONDAY_OPEN,
        force=True,
    )
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    reasons = [event["reason"] for event in events if event["event"] == "session_skipped"]
    assert len(reasons) == 3
    assert any("marché fermé" in reason for reason in reasons)
    assert any("budget" in reason for reason in reasons)
    assert any("pause auto" in reason for reason in reasons)


def test_token_budget_blocks_the_session(tmp_path):
    state = AgentState(agent_id="test", tokens_spent_today=200_000, budget_day="2026-10-26")
    outcome = run_session(
        MockBroker(cash=1080.0), BuyThenStop(), state, JsonlLogger(tmp_path / "a.jsonl"),
        now=MONDAY_OPEN, force=True, budget_tokens=120_000,
    )
    assert not outcome.ran
    assert "budget" in outcome.reason


def test_three_consecutive_tool_failures_pause_the_agent(tmp_path):
    state = AgentState(agent_id="test", consecutive_tool_failures=3)
    outcome = run_session(
        MockBroker(cash=1080.0), BuyThenStop(), state, JsonlLogger(tmp_path / "a.jsonl"),
        now=MONDAY_OPEN, force=True,
    )
    assert not outcome.ran
    assert "pause auto" in outcome.reason


def test_self_critique_runs_every_five_trades(tmp_path):
    state = AgentState(agent_id="test", trades_since_critique=5)
    client = CritiqueClient()
    outcome = run_session(
        MockBroker(cash=1080.0),
        client,
        state,
        JsonlLogger(tmp_path / "a.jsonl"),
        now=MONDAY_OPEN,
        force=True,
    )
    assert outcome.critique is not None
    assert "Plan" in state.plan
    assert state.trades_since_critique == 0
    assert state.critique_count == 1


def test_invalid_arguments_are_counted_as_tool_failures(tmp_path):
    bad = ToolCall(id="x", name="place_order", arguments={"symbol": "SPY"}, raw_arguments="{}")
    state = AgentState(agent_id="test")
    run_session(
        MockBroker(cash=1080.0), BuyThenStop(bad), state, JsonlLogger(tmp_path / "a.jsonl"),
        now=MONDAY_OPEN, force=True,
    )
    assert state.consecutive_tool_failures >= 1
    assert state.total_trades == 0


# -- état persistant ------------------------------------------------------------

def test_state_survives_a_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    state = AgentState(agent_id="a", total_trades=7, plan="rester prudent")
    state.mark_slot_done("2026-10-26:open")
    save_state(path, state)
    reloaded = load_state(path, "a")
    assert reloaded.total_trades == 7
    assert reloaded.plan == "rester prudent"
    assert reloaded.is_slot_done("2026-10-26:open")


def test_corrupt_state_is_quarantined_not_crashing(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    state = load_state(path, "a")
    assert state.total_trades == 0
    assert (tmp_path / "state.json.corrupt").exists()


def test_day_rollover_resets_daily_tokens():
    state = AgentState(agent_id="a", tokens_spent_today=5000, budget_day="2026-10-24")
    state.roll_day(datetime(2026, 10, 25, tzinfo=UTC).date())
    assert state.tokens_spent_today == 0


@pytest.mark.parametrize("qty", [1, 2.5, 10])
def test_orders_are_idempotent_at_the_broker(qty):
    from tradingbench.harness.executor import ToolExecutor

    broker = MockBroker(cash=1080.0)
    executor = ToolExecutor(broker, client_order_prefix="agent-slot")
    first = json.loads(
        executor.dispatch("place_order", {"symbol": "SPY", "qty": qty, "side": "buy"})
    )
    assert first["status"] == "filled"
    # Rejouer exactement le même identifiant ne doit pas créer un second ordre.
    order = broker.place_order("SPY", qty, side=Side.BUY, client_order_id="agent-slot-0")
    assert order.id[:8] == first["id"]


# -- capital réaliste (1 000 €) -------------------------------------------------

def test_an_overfunded_account_is_flagged_but_not_blocking(tmp_path):
    """Alpaca n'a pas d'API de reset : on signale et on plafonne l'exposition au lieu
    de bloquer la boucle."""
    log_path = tmp_path / "a.jsonl"
    outcome = run_session(
        MockBroker(cash=100_000.0),
        BuyThenStop(),
        AgentState(agent_id="test"),
        JsonlLogger(log_path),
        now=MONDAY_OPEN,
        force=True,
    )
    assert outcome.ran
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert any(event["event"] == "capital_warning" for event in events)


def test_a_buy_beyond_the_budget_is_refused():
    import json as _json

    from tradingbench.harness.executor import ToolExecutor

    broker = MockBroker(cash=100_000.0)
    broker.set_price("SPY", 500.0)
    executor = ToolExecutor(broker, max_notional=1080.0)
    payload = _json.loads(
        executor.dispatch("place_order", {"symbol": "SPY", "qty": 10, "side": "buy"})
    )
    assert "dépassement du capital" in payload["error"]
    assert executor.order_results == []


def test_a_buy_within_the_budget_goes_through():
    import json as _json

    from tradingbench.harness.executor import ToolExecutor

    broker = MockBroker(cash=1080.0)
    broker.set_price("SPY", 100.0)
    executor = ToolExecutor(broker, max_notional=1080.0)
    payload = _json.loads(
        executor.dispatch("place_order", {"symbol": "SPY", "qty": 5, "side": "buy"})
    )
    assert payload["status"] == "filled"


def test_a_compliant_account_records_its_starting_equity(tmp_path):
    state = AgentState(agent_id="test")
    run_session(
        MockBroker(cash=1080.0),
        BuyThenStop(),
        state,
        JsonlLogger(tmp_path / "a.jsonl"),
        now=MONDAY_OPEN,
        force=True,
    )
    assert state.starting_equity_usd == 1080.0


def test_the_prompt_states_the_real_capital_and_protection_rule():
    prompt = format_system_prompt()
    assert "1 000 €" in prompt
    assert "stop_loss" in prompt and "take_profit" in prompt
    assert "NON FIABLES" in prompt  # consigne d'injection de prompt


# -- actualités et veille -------------------------------------------------------

def test_news_are_injected_and_flagged_untrusted():
    context = build_news_context(MockBroker())
    assert "AAPL" in context and "résultats" in context


def test_news_failure_does_not_break_the_session():
    class NoNews(MockBroker):
        def get_news(self, symbols, *, limit=5):
            raise RuntimeError("news API down")

    assert "indisponibles" in build_news_context(NoNews())


def test_veille_is_read_with_its_age(tmp_path):
    from tradingbench.agent.session import read_veille

    file = tmp_path / "veille.md"
    file.write_text("# Veille\n- TSMC publie de bons résultats", encoding="utf-8")
    text = read_veille(file, now=datetime.now(UTC))
    assert "âge 0.0 jour" in text
    assert "TSMC" in text


def test_stale_veille_is_flagged(tmp_path):
    from tradingbench.agent.session import read_veille

    file = tmp_path / "veille.md"
    file.write_text("vieille veille", encoding="utf-8")
    old = datetime.now(UTC).timestamp() - 10 * 86400
    os.utime(file, (old, old))
    assert "PÉRIMÉE" in read_veille(file)


def test_veille_directory_picks_the_most_recent_file(tmp_path):
    from tradingbench.agent.session import read_veille

    (tmp_path / "vieux.md").write_text("vieux", encoding="utf-8")
    recent = tmp_path / "recent.md"
    recent.write_text("récent", encoding="utf-8")
    old = datetime.now(UTC).timestamp() - 5 * 86400
    os.utime(tmp_path / "vieux.md", (old, old))
    assert "récent" in read_veille(tmp_path)


def test_missing_veille_is_explicit_not_silent(tmp_path):
    from tradingbench.agent.session import read_veille

    assert "n'existe pas" in read_veille(tmp_path / "absent.md")
    assert "aucune veille configurée" in read_veille(None)
