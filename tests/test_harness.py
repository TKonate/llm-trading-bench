"""The harness must be strict: a wrong argument is a failure, not a nuance."""

from __future__ import annotations

import pytest

from tradingbench.harness.scenarios import SCENARIOS, Scenario, score
from tradingbench.llm.client import Completion, ToolCall


def call(name: str, **args) -> ToolCall:
    import json

    return ToolCall(id="1", name=name, arguments=args, raw_arguments=json.dumps(args))


def by_id(scenario_id: str) -> Scenario:
    return next(s for s in SCENARIOS if s.id == scenario_id)


def test_correct_call_passes():
    outcome = score(by_id("buy"), [call("place_order", symbol="SPY", qty=3, side="buy")])
    assert outcome.ok, outcome.reason


def test_side_is_case_insensitive():
    assert score(by_id("buy"), [call("place_order", symbol="spy", qty=3, side="BUY")]).ok


def test_wrong_side_fails():
    outcome = score(by_id("sell"), [call("place_order", symbol="NVDA", qty=2, side="buy")])
    assert not outcome.ok
    assert "side" in outcome.reason


def test_wrong_quantity_fails():
    outcome = score(by_id("buy"), [call("place_order", symbol="SPY", qty=30, side="buy")])
    assert not outcome.ok
    assert "qty" in outcome.reason


def test_missing_argument_fails():
    outcome = score(by_id("buy"), [call("place_order", symbol="SPY", side="buy")])
    assert not outcome.ok
    assert "manquant" in outcome.reason


def test_wrong_tool_fails():
    outcome = score(by_id("quote"), [call("get_portfolio")])
    assert not outcome.ok
    assert "mauvais outil" in outcome.reason


def test_no_call_fails_when_a_tool_is_expected():
    outcome = score(by_id("quote"), [])
    assert not outcome.ok
    assert "aucun outil" in outcome.reason


def test_forbidden_tool_fails_even_with_other_valid_calls():
    outcome = score(
        by_id("hold"),
        [call("get_quote", symbol="SPY"), call("place_order", symbol="SPY", qty=1, side="buy")],
    )
    assert not outcome.ok
    assert "interdit" in outcome.reason


def test_invalid_json_arguments_fail():
    broken = ToolCall(id="1", name="place_order", arguments={}, raw_arguments="{not json")
    outcome = score(by_id("buy"), [broken])
    assert not outcome.ok
    assert "illisibles" in outcome.reason


def test_every_scenario_declares_why_it_exists():
    assert all(s.why for s in SCENARIOS), "chaque scénario doit documenter son intention"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_has_a_concrete_prompt(scenario):
    assert len(scenario.prompt) > 10
    assert scenario.expect_tool or scenario.forbid_tool, "scénario sans critère = scénario inutile"


class ScriptedClient:
    """Faux modèle : renvoie des appels préparés, scénario par scénario."""

    model = "scripted"

    def __init__(self, script: dict[str, list[ToolCall]]):
        self.script = script
        self.seen: list[str] = []

    def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=800):
        prompt = messages[-1]["content"]
        self.seen.append(prompt)
        calls = self.script.get(prompt, [])
        return Completion(
            text="",
            tool_calls=calls,
            model=self.model,
            finish_reason="tool_calls" if calls else "stop",
            latency_ms=1.0,
        )


def test_runner_scores_a_perfect_model_as_go():
    from tradingbench.harness.runner import run_scenarios, summarize

    script = {
        scenario.prompt: (
            [call(scenario.expect_tool, **scenario.expect_args)] if scenario.expect_tool else []
        )
        for scenario in SCENARIOS
    }
    results = run_scenarios(ScriptedClient(script))
    summary = summarize(results)[0]
    assert summary.passed == summary.total, [r.reason for r in results if not r.ok]
    assert summary.verdict == "GO"


def test_runner_scores_a_broken_model_as_no_go():
    from tradingbench.harness.runner import run_scenarios, summarize

    results = run_scenarios(ScriptedClient({}))  # ne répond jamais avec un outil
    summary = summarize(results)[0]
    assert summary.passed < summary.total
    assert summary.verdict == "NO-GO"


def test_runner_counts_a_crashing_model_as_failure_not_exception():
    from tradingbench.harness.runner import run_scenarios

    class Broken:
        model = "broken"

        def complete(self, *args, **kwargs):
            raise RuntimeError("provider down")

    results = run_scenarios(Broken(), SCENARIOS[:2])
    assert all(not result.ok for result in results)
    assert all(result.error and "provider down" in result.error for result in results)


class TwoStepClient:
    """Modèle réaliste : consulte le prix, puis décide. C'est ce que fait un agent."""

    model = "twostep"

    def __init__(self) -> None:
        self.turn = 0

    def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=800):
        self.turn += 1
        if self.turn == 1:
            return Completion(
                text="",
                tool_calls=[call("get_quote", symbol="SPY")],
                model=self.model,
                finish_reason="tool_calls",
                latency_ms=1.0,
            )
        if self.turn == 2:
            return Completion(
                text="",
                tool_calls=[call("place_order", symbol="SPY", qty=3, side="buy")],
                model=self.model,
                finish_reason="tool_calls",
                latency_ms=1.0,
            )
        return Completion(
            text="Ordre passé.",
            tool_calls=[],
            model=self.model,
            finish_reason="stop",
            latency_ms=1.0,
        )


def test_a_model_that_checks_the_price_before_ordering_passes():
    """Régression : en mono-tour, ce comportement — correct — était compté FAIL."""
    from tradingbench.harness.runner import run_scenarios

    results = run_scenarios(TwoStepClient(), (by_id("buy"),))
    assert results[0].ok, results[0].reason
    assert results[0].called == ["get_quote", "place_order"]


def test_the_tool_loop_is_capped():
    from tradingbench.harness.runner import run_scenarios

    class Endless:
        model = "endless"

        def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=800):
            return Completion(
                text="",
                tool_calls=[call("get_quote", symbol="SPY")],
                model=self.model,
                finish_reason="tool_calls",
                latency_ms=1.0,
            )

    results = run_scenarios(Endless(), (by_id("quote"),), max_turns=3)
    assert len(results[0].called) == 3  # borné, pas de boucle infinie


# -- exécution réelle des outils ------------------------------------------------

def test_executor_returns_structured_market_data():
    import json

    from tradingbench.harness.executor import ToolExecutor

    quote = json.loads(ToolExecutor().dispatch("get_quote", {"symbol": "spy"}))
    assert quote["symbol"] == "SPY"
    assert quote["ask"] >= quote["bid"] > 0


def test_executor_places_a_real_order_on_the_mock_broker():
    import json

    from tradingbench.harness.executor import ToolExecutor

    executor = ToolExecutor()
    result = json.loads(
        executor.dispatch("place_order", {"symbol": "SPY", "qty": 1, "side": "buy"})
    )
    assert result["status"] == "filled"
    assert executor.order_attempts == [
        {"symbol": "SPY", "qty": 1.0, "side": "buy", "stop_loss": None, "take_profit": None}
    ]


def test_executor_reports_broker_errors_without_raising():
    import json

    from tradingbench.harness.executor import ToolExecutor

    payload = json.loads(ToolExecutor().dispatch("get_quote", {"symbol": "INCONNU"}))
    assert "error" in payload


def test_executor_rejects_an_unknown_tool():
    import json

    from tradingbench.harness.executor import ToolExecutor

    payload = json.loads(ToolExecutor().dispatch("delete_everything", {}))
    assert "outil inconnu" in payload["error"]


# -- garde-fou des protections --------------------------------------------------

def test_a_stop_loss_above_the_price_is_refused():
    import json

    from tradingbench.brokers.mock import MockBroker
    from tradingbench.harness.executor import ToolExecutor

    broker = MockBroker()
    price = broker.get_quote("SPY").mid
    payload = json.loads(
        ToolExecutor(broker).dispatch(
            "place_order",
            {"symbol": "SPY", "qty": 1, "side": "buy", "stop_loss": price + 10},
        )
    )
    assert "error" in payload and "stop_loss" in payload["error"]


def test_a_take_profit_below_the_price_is_refused():
    import json

    from tradingbench.brokers.mock import MockBroker
    from tradingbench.harness.executor import ToolExecutor

    broker = MockBroker()
    price = broker.get_quote("SPY").mid
    payload = json.loads(
        ToolExecutor(broker).dispatch(
            "place_order",
            {"symbol": "SPY", "qty": 1, "side": "buy", "take_profit": price - 10},
        )
    )
    assert "error" in payload and "take_profit" in payload["error"]


def test_a_valid_bracket_goes_through_and_is_recorded():
    import json

    from tradingbench.brokers.mock import MockBroker
    from tradingbench.harness.executor import ToolExecutor

    broker = MockBroker()
    price = broker.get_quote("SPY").mid
    executor = ToolExecutor(broker)
    payload = json.loads(
        executor.dispatch(
            "place_order",
            {
                "symbol": "SPY",
                "qty": 1,
                "side": "buy",
                "stop_loss": round(price * 0.95, 2),
                "take_profit": round(price * 1.10, 2),
            },
        )
    )
    assert payload["status"] == "filled"
    assert executor.order_attempts[0]["stop_loss"] is not None
    assert executor.order_attempts[0]["take_profit"] is not None


def test_get_news_tool_returns_headlines_and_a_trust_warning():
    import json

    from tradingbench.brokers.mock import MockBroker
    from tradingbench.harness.executor import ToolExecutor

    payload = json.loads(ToolExecutor(MockBroker()).dispatch("get_news", {"symbols": ["AAPL"]}))
    assert payload["news"][0]["headline"]
    assert "n" in payload["note"] and "instructions" in payload["note"]
