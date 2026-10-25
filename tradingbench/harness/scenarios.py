"""Scénarios de test du tool calling — le filtre éliminatoire du benchmark.

Pourquoi ce harnais existe : dans la vidéo Defend Intelligence, Mistral a passé
une semaine à échouer ses appels d'outils (−7 % de performance, en pure friction
d'exécution). Un modèle incapable d'appeler correctement ses outils n'est pas un
concurrent : c'est du bruit dans les résultats.

Chaque scénario teste un mode de défaillance observé dans la réalité :
appel manquant, mauvais outil, arguments inventés, garde-fou ignoré.
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
        "place_order",
        "Passer un ordre au marché ou à cours limité.",
        {
            "symbol": {"type": "string"},
            "qty": {"type": "number", "description": "Nombre d'actions, > 0"},
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "order_type": {"type": "string", "enum": ["market", "limit"]},
        },
        ["symbol", "qty", "side"],
    ),
]


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    prompt: str
    expect_tool: str | None = None
    expect_args: dict[str, Any] = field(default_factory=dict)
    forbid_tool: str | None = None
    why: str = ""
    seed: tuple[tuple[str, float], ...] = ()
    """Positions ouvertes avant le scénario. Une vente depuis un portefeuille vide
    teste la chance du modèle, pas sa compétence."""


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
    ok: bool
    reason: str


def _args_match(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, str]:
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
    """Évalue une réponse. Volontairement strict : un faux positif ici fausse tout le classement."""
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
