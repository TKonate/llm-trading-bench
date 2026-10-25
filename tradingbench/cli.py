"""Command line entry point: `tradingbench doctor` / `tradingbench smoke`."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import __version__
from .brokers import BrokerError, get_broker
from .config import Settings
from .jsonlog import JsonlLogger
from .models import OrderType, Side

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_ENV_LOADED = False


def _load_env() -> str | None:
    """Charge .env sans jamais faire planter le CLI s'il est illisible.

    Cas réel : `docker exec` tourne en root, écrit .env en root:600, et
    l'utilisateur du projet ne peut plus le lire. On doit le dire clairement,
    pas lever une PermissionError au moment de l'import.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return None
    _ENV_LOADED = True
    try:
        load_dotenv(ENV_FILE)
    except OSError as exc:
        return str(exc)
    return None


app = typer.Typer(add_completion=False, help="LLM paper-trading benchmark toolbox.")
console = Console()


def _settings() -> Settings:
    problem = _load_env()
    if problem:
        console.print(f"[yellow].env illisible :[/yellow] {problem}")
        console.print(
            "[yellow]→[/yellow] docker exec hermes chown 1001:1001 "
            "/opt/data/projects/llm-trading-bench/.env"
        )
    return Settings.from_env()


@dataclass
class Check:
    name: str
    ok: bool
    ms: float
    detail: str


def _run(
    name: str,
    func: Callable[[], Any],
    describe: Callable[[Any], str],
    validate: Callable[[Any], bool] | None = None,
) -> Check:
    """Run one check. `validate` matters: an HTTP 200 with an empty payload is
    not a success, and reporting it as one is how broken pipelines ship."""
    started = time.perf_counter()
    try:
        value = func()
        elapsed = (time.perf_counter() - started) * 1000
        ok = True if validate is None else bool(validate(value))
        detail = describe(value)
        if not ok:
            detail = f"réponse inattendue — {detail}"
        return Check(name, ok, elapsed, detail)
    except Exception as exc:  # noqa: BLE001 - a smoke test must report, not crash
        elapsed = (time.perf_counter() - started) * 1000
        return Check(name, False, elapsed, f"{type(exc).__name__}: {exc}")


@app.command()
def setkeys(
    verify: bool = typer.Option(True, help="teste les clés juste après les avoir écrites"),
) -> None:
    """Enregistrer les clés Alpaca paper dans .env (saisie masquée, sans éditeur)."""
    from . import setkeys as setkeys_module

    raise typer.Exit(code=setkeys_module.run(verify=verify))


@app.command()
def doctor(broker: str = typer.Option("mock", help="mock | alpaca")) -> None:
    """Show configuration state without touching the network."""
    settings = _settings()
    table = Table(
        title=f"tradingbench {__version__} — configuration",
        show_header=False,
    )
    table.add_column("clé", style="cyan")
    table.add_column("valeur")
    table.add_row("broker demandé", broker)
    table.add_row("ALPACA_API_KEY", "présent" if settings.alpaca_api_key else "[red]absent[/red]")
    table.add_row(
        "ALPACA_SECRET_KEY", "présent" if settings.alpaca_secret_key else "[red]absent[/red]"
    )
    table.add_row("feed", settings.alpaca_feed)
    table.add_row("logs", str(settings.log_dir))
    console.print(table)
    if broker.startswith("alpaca") and not settings.has_alpaca_credentials:
        console.print(
            "\n[yellow]Compte paper gratuit :[/yellow] https://app.alpaca.markets → "
            "Paper Trading → API Keys. Puis renseigner .env "
            "(ne jamais coller les clés dans un chat)."
        )
        raise typer.Exit(code=2)


@app.command()
def smoke(
    broker: str = typer.Option("mock", help="mock | alpaca"),
    symbol: str = typer.Option("SPY", help="symbole de test"),
    place_order: bool = typer.Option(
        False, "--place-order", help="teste le chemin d'ordre (limit hors marché + annulation)"
    ),
) -> None:
    """End-to-end connectivity check against the selected broker."""
    settings = _settings()
    logger = JsonlLogger(settings.log_dir / "smoke.jsonl")
    checks: list[Check] = []

    try:
        client = get_broker(broker, settings)
    except BrokerError as exc:
        console.print(f"[red]Impossible d'instancier le broker :[/red] {exc}")
        raise typer.Exit(code=2) from exc

    checks.append(_run("market clock", client.is_market_open, lambda v: f"open={v}"))
    checks.append(
        _run(
            "account",
            client.get_account,
            lambda a: f"{a.account_id} · cash {a.cash:,.2f} {a.currency} · equity {a.equity:,.2f}",
        )
    )
    checks.append(
        _run(
            f"quote {symbol}",
            lambda: client.get_quote(symbol),
            lambda q: f"bid {q.bid:.2f} / ask {q.ask:.2f} · spread {q.spread_bps:.1f} bps",
        )
    )
    checks.append(
        _run(
            f"bars {symbol}",
            lambda: client.get_bars(symbol, limit=10),
            lambda bars: f"{len(bars)} barres · dernière clôture "
            f"{bars[-1].close:.2f} ({bars[-1].timestamp.date()})" if bars else "0 barre",
            validate=lambda bars: len(bars) >= 2,
        )
    )
    checks.append(
        _run(
            "positions",
            client.get_positions,
            lambda positions: f"{len(positions)} position(s)",
        )
    )

    if place_order:
        quote = client.get_quote(symbol)

        def _roundtrip() -> str:
            order = client.place_order(
                symbol,
                1,
                Side.BUY,
                order_type=OrderType.LIMIT,
                limit_price=round(quote.mid * 0.5, 2),
            )
            client.cancel_order(order.id)
            final = client.get_order(order.id)
            return f"ordre {order.id[:8]}… {order.status.value} → {final.status.value}"

        checks.append(_run("order roundtrip", _roundtrip, lambda detail: detail))

    table = Table(title=f"smoke test — {client.name}", show_lines=False)
    table.add_column("check")
    table.add_column("état", justify="center")
    table.add_column("ms", justify="right")
    table.add_column("détail")
    for check in checks:
        table.add_row(
            check.name,
            "[green]OK[/green]" if check.ok else "[red]FAIL[/red]",
            f"{check.ms:.0f}",
            check.detail,
        )
    console.print(table)

    logger.log(
        "smoke",
        broker=client.name,
        ok=all(c.ok for c in checks),
        checks=[
            {"name": c.name, "ok": c.ok, "ms": round(c.ms, 1), "detail": c.detail}
            for c in checks
        ],
    )
    console.print(f"[dim]log → {logger.path}[/dim]")

    if not all(c.ok for c in checks):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
