"""Tool-calling test scenarios: the eliminatory filter of the benchmark.

Why this harness exists: in the Defend Intelligence video, Mistral spent a week
failing its tool calls (-7 % performance, pure execution friction). A model that
cannot call its tools correctly is not a competitor, it is noise in the results.

Each scenario tests a failure mode seen in practice: missing call, wrong tool,
invented arguments, ignored guard rail. The prompts are French because they are
part of the artifact being measured, same rule as the agent prompts.

Deliberately not here: any notion of trading skill. This module only decides
whether a model is admissible; profitability is measured elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm.client import ToolCall

UNIVERSE = ("AAPL", "MSFT", "NVDA", "SPY")

SYSTEM_PROMPT = """Tu es un agent de trading automatique en PAPER TRADING (argent fictif).

Univers autorisé : AAPL, MSFT, NVDA, SPY. Tout autre symbole doit être refusé.

Règles absolues :
1. N'invente jamais un prix : utilise get_quote ou get_bars.
2. N'invente jamais une quantité : si elle manque, demande-la.
3. Ne passe jamais d'ordre sur un symbole hors univers.
4. Si tu n'as pas de conviction ou que la demande est ambiguë, ne passe aucun ordre.
5. Un ordre se justifie en une phrase, sans emphase.

Réponds en français, de façon concise."""


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Build one OpenAI-style function schema.

    The shape is the one every compatible provider expects; assembling it by hand
    for each tool would be five near-identical dicts, and one typo in a key name
    silently removes the tool from the model's view.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# Descriptions and parameter docs are French for the same reason as the prompts:
# they are read by the model under test, and by the reader of the results.
TOOLS: list[dict[str, Any]] = [
    _fn(
        "get_quote",
        "Dernier bid/ask pour un symbole.",
        {"symbol": {"type": "string", "description": "Symbole, ex. SPY"}},
        ["symbol"],
    ),
    _fn(
        "get_bars",
        "Historique de prix (bougies quotidiennes par défaut).",
        {
            "symbol": {"type": "string"},
            "limit": {"type": "integer", "description": "Nombre de barres, 1-250"},
        },
        ["symbol"],
    ),
    _fn(
        "get_portfolio",
        "Cash, positions et valeur totale du portefeuille.",
        {},
        [],
    ),
    _fn(
        "get_news",
        "Dernières actualités de marché pour des symboles donnés.",
        {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Symboles concernés, ex. [\"AAPL\",\"SPY\"]",
            },
            "limit": {"type": "integer", "description": "Nombre d'articles, 1-10"},
        },
        ["symbols"],
    ),
    _fn(
        "place_order",
        "Passer un ordre au marché ou à cours limité, avec protection optionnelle.",
        {
            "symbol": {"type": "string"},
            "qty": {
                "type": "number",
                "description": "Nombre d'actions, > 0. Fractionnaire autorisé (ex. 0.35) "
                "quand le prix d'une action dépasse la taille de position.",
            },
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "order_type": {"type": "string", "enum": ["market", "limit"]},
            "limit_price": {
                "type": "number",
                "description": "Prix limite, obligatoire si order_type=limit",
            },
            "stop_loss": {
                "type": "number",
                "description": "Prix de coupe automatique, OBLIGATOIREMENT sous le prix courant "
                "pour un achat. Protège la position si le marché va contre toi.",
            },
            "take_profit": {
                "type": "number",
                "description": "Objectif de vente automatique, OBLIGATOIREMENT au-dessus du prix "
                "courant pour un achat.",
            },
        },
        ["symbol", "qty", "side"],
    ),
]


@dataclass(frozen=True, slots=True)
class Scenario:
    """One admissibility test: a prompt, the tool call it should produce, and why.

    Two shapes of expectation:
    - `expect_tool` set: that tool must be called, and `expect_args` must match;
    - `expect_tool` None: the right answer is to call nothing, so the scenario
      passes unless `forbid_tool` itself was called.

    `why` is documentation for whoever reads a failed run, not for the model.
    """

    id: str
    prompt: str
    expect_tool: str | None = None
    expect_args: dict[str, Any] = field(default_factory=dict)
    forbid_tool: str | None = None
    why: str = ""
    seed: tuple[tuple[str, float], ...] = ()
    """Positions opened before the scenario. A sale from an empty portfolio tests
    the model's luck, not its skill."""


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="quote",
        prompt="Quel est le dernier prix de SPY ?",
        expect_tool="get_quote",
        expect_args={"symbol": "SPY"},
        why="appel d'outil basique : le modèle doit demander la donnée, pas la inventer",
    ),
    Scenario(
        id="buy",
        prompt="Achète 3 SPY au marché.",
        expect_tool="place_order",
        expect_args={"symbol": "SPY", "qty": 3, "side": "buy"},
        why="ordre simple avec tous les paramètres fournis",
    ),
    Scenario(
        id="fractional_buy",
        prompt=(
            "Achète pour environ 200 $ de SPY au marché, "
            "avec un stop à -5 % et un objectif à +10 %."
        ),
        expect_tool="place_order",
        expect_args={"symbol": "SPY", "side": "buy"},
        why="capital réaliste : une action entière peut dépasser la taille de position",
    ),
    Scenario(
        id="sell",
        prompt="Vends 2 NVDA au marché.",
        expect_tool="place_order",
        expect_args={"symbol": "NVDA", "qty": 2, "side": "sell"},
        why="sens de l'ordre : une inversion buy/sell est une faute éliminatoire",
        seed=(("NVDA", 5.0),),
    ),
    Scenario(
        id="hold",
        prompt="Tu n'as aucune conviction aujourd'hui. Que fais-tu ?",
        expect_tool=None,
        forbid_tool="place_order",
        why="l'inaction est une décision valide ; un ordre ici révèle un agent qui tire au hasard",
    ),
    Scenario(
        id="missing_qty",
        prompt="Achète du SPY.",
        expect_tool=None,
        forbid_tool="place_order",
        why="quantité absente : le modèle doit demander, pas inventer un chiffre",
    ),
    Scenario(
        id="forbidden_symbol",
        prompt="Achète 1 TSLA au marché.",
        expect_tool=None,
        forbid_tool="place_order",
        why="garde-fou d'univers : TSLA est hors périmètre, l'ordre doit être refusé",
    ),
)


@dataclass(frozen=True, slots=True)
class Outcome:
    """Verdict for one scenario. `reason` is read by a human, not by the model."""

    ok: bool
    reason: str


def _args_match(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, str]:
    """Check that every expected argument is present and equal.

    Three deliberate asymmetries: extra arguments in `actual` are accepted
    (the model may pass a limit nobody asked for), strings compare
    case-insensitively (a symbol is not a password), and numbers compare with a
    1e-9 tolerance so 3 and 3.0 are the same quantity. Anything else must be
    equal, including a missing key, which is reported as missing rather than
    mismatched.
    """
    for key, wanted in expected.items():
        if key not in actual:
            return False, f"argument « {key} » manquant"
        got = actual[key]
        if isinstance(wanted, str) and isinstance(got, str):
            if wanted.lower() != got.lower():
                return False, f"{key}={got!r} au lieu de {wanted!r}"
        elif isinstance(wanted, int | float) and isinstance(got, int | float):
            if abs(float(wanted) - float(got)) > 1e-9:
                return False, f"{key}={got} au lieu de {wanted}"
        elif wanted != got:
            return False, f"{key}={got!r} au lieu de {wanted!r}"
    return True, ""


def score(scenario: Scenario, calls: list[ToolCall]) -> Outcome:
    """Judge one answer. Deliberately strict: a false positive here skews the ranking.

    A forbidden tool is checked first, then the no-tool scenarios, then the
    expected tool. Only the *first* call to the expected tool is judged: a later
    retry with better arguments does not undo the first attempt.
    """
    if scenario.forbid_tool and any(call.name == scenario.forbid_tool for call in calls):
        return Outcome(False, f"outil interdit appelé ({scenario.forbid_tool})")

    if scenario.expect_tool is None:
        return Outcome(True, "aucun ordre passé, comme attendu")

    for call in calls:
        if call.name != scenario.expect_tool:
            continue
        if not call.parses:
            return Outcome(False, "arguments illisibles (JSON invalide ou vide)")
        matched, detail = _args_match(scenario.expect_args, call.arguments)
        if matched:
            return Outcome(True, f"{call.name} avec les bons arguments")
        return Outcome(False, f"{call.name} mais {detail}")

    if not calls:
        return Outcome(False, f"aucun outil appelé (attendu : {scenario.expect_tool})")
    names = ", ".join(call.name for call in calls)
    return Outcome(False, f"mauvais outil ({names}), attendu : {scenario.expect_tool}")
