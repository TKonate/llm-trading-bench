"""État persistant d'un agent : ce qui doit survivre à un redémarrage.

Trois choses ne doivent jamais être perdues : la session déjà exécutée (sinon on
rejoue des ordres), le compteur d'échecs consécutifs (sinon la pause automatique
ne se déclenche jamais) et la consommation de tokens du jour (sinon le plafond de
budget est décoratif).

L'écriture est atomique (fichier temporaire + remplacement) : un arrêt brutal au
mauvais moment ne doit pas laisser un JSON tronqué.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AgentState:
    agent_id: str
    prompt_version: str = ""
    slots_done: list[str] = field(default_factory=list)
    total_trades: int = 0
    trades_since_critique: int = 0
    consecutive_tool_failures: int = 0
    tokens_spent_today: int = 0
    budget_day: str = ""
    plan: str = ""
    critique_count: int = 0
    recent_trades: list[dict[str, Any]] = field(default_factory=list)
    starting_equity_usd: float | None = None

    # -- sérialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "prompt_version": self.prompt_version,
            "slots_done": self.slots_done[-14:],
            "total_trades": self.total_trades,
            "trades_since_critique": self.trades_since_critique,
            "consecutive_tool_failures": self.consecutive_tool_failures,
            "tokens_spent_today": self.tokens_spent_today,
            "budget_day": self.budget_day,
            "plan": self.plan,
            "critique_count": self.critique_count,
            "recent_trades": self.recent_trades[-20:],
            "starting_equity_usd": self.starting_equity_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        return cls(**{key: data[key] for key in cls.__slots__ if key in data})  # type: ignore[arg-type]

    # -- helpers ---------------------------------------------------------
    def is_slot_done(self, slot: str) -> bool:
        return slot in self.slots_done

    def mark_slot_done(self, slot: str) -> None:
        if slot not in self.slots_done:
            self.slots_done.append(slot)

    def roll_day(self, today: date | None = None) -> None:
        today = today or datetime.now(UTC).date()
        if self.budget_day != today.isoformat():
            self.budget_day = today.isoformat()
            self.tokens_spent_today = 0

    def record_trade(self, trade: dict[str, Any]) -> None:
        self.total_trades += 1
        self.trades_since_critique += 1
        self.recent_trades.append(trade)

    def record_call(self, tokens: int, tool_failures: int) -> None:
        self.tokens_spent_today += tokens
        self.consecutive_tool_failures = (
            self.consecutive_tool_failures + tool_failures if tool_failures else 0
        )


def load_state(path: Path, agent_id: str) -> AgentState:
    if not path.exists():
        return AgentState(agent_id=agent_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # État corrompu : on repart propre, mais on garde le fichier pour enquête.
        path.rename(path.with_suffix(path.suffix + ".corrupt"))
        return AgentState(agent_id=agent_id)
    data.setdefault("agent_id", agent_id)
    return AgentState.from_dict(data)


def save_state(path: Path, state: AgentState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
