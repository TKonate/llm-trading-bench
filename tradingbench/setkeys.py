"""Write Alpaca paper keys into `.env` without an editor and without a chat.

Usage: `tradingbench setkeys`, from a shell where the venv is active.

Values are read with `getpass` (no echo), written into `.env` while preserving
comments, then the file is set to mode 600, and finally a live connection test
runs: a key that "looks right" is not a key that works.

Deliberately no keyring, no encryption and no backup copy: the point of the
mode-600 `.env` is that the secret has exactly one home on this machine.
"""

from __future__ import annotations

import getpass
import os
import stat
from pathlib import Path

from dotenv import set_key, unset_key

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def mask(value: str) -> str:
    """Render a secret as enough to recognise it, never enough to reuse it.

    Eight characters or fewer are masked entirely: keeping a prefix and a
    suffix would leave most of a short key on screen.
    """
    if not value:
        return "(vide)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…{value[-4:]} ({len(value)} car.)"


def update_env_file(path: Path, values: dict[str, str]) -> None:
    """Write key/value pairs into a .env file without losing its comments.

    An empty value means "remove the key" rather than "set it to empty", so a
    cleared credential cannot survive as a blank line that later looks set.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    for key, value in values.items():
        if value:
            set_key(str(path), key, value)
        else:
            unset_key(str(path), key)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
    _match_project_owner(path)


def _match_project_owner(path: Path) -> None:
    """Re-own the file to the project's user when running as root (docker exec).

    Without this, .env belongs to root:root in 600 and every tool running as the
    project user (smoke test, decision loop) loses the ability to read it.
    """
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    info = path.parent.stat()
    os.chown(path, info.st_uid, info.st_gid)


def _prompt_secret(label: str) -> str:
    """Read one secret from the terminal, refusing anything that looks truncated.

    getpass raises EOFError when stdin is not a TTY, which is the common case
    behind a `docker exec` without `-i`; the message says so instead of letting
    an empty key be written.
    """
    try:
        value = getpass.getpass(f"{label} : ")
    except (EOFError, OSError):
        raise SystemExit(
            "Entrée standard non interactive. Relance avec `docker exec -it …` "
            "(le -i est indispensable)."
        ) from None
    value = value.strip()
    if not value:
        raise SystemExit("Valeur vide — rien n'a été enregistré.")
    if any(ch.isspace() for ch in value):
        raise SystemExit("La valeur contient un espace — copie probablement incomplète.")
    return value


def run(verify: bool = True) -> int:
    """Prompt for both keys, persist them, and optionally prove they work.

    Returns the process exit code: 0 when the keys were written (and verified),
    1 when the connection test failed. The keys are kept in that case, so the
    user can fix the account without re-typing them.
    """
    print("Clés API Alpaca PAPER (https://app.alpaca.markets → Paper Trading → API Keys)")
    print("Ce que tu tapes ne s'affiche pas et n'est envoyé nulle part.\n")
    api_key = _prompt_secret("ALPACA_API_KEY")
    secret_key = _prompt_secret("ALPACA_SECRET_KEY")

    update_env_file(ENV_PATH, {"ALPACA_API_KEY": api_key, "ALPACA_SECRET_KEY": secret_key})
    print(f"\n✓ Écrites dans {ENV_PATH} (droits 600)")
    print(f"  API key    : {mask(api_key)}")
    print(f"  Secret key : {mask(secret_key)}")
    print("  ↳ le secret est enregistré mais volontairement masqué, même ici.")

    if not verify:
        return 0

    print("\nTest de connexion au compte paper…")
    try:
        # Deferred import: writing keys needs no broker, and httpx plus the
        # broker stack should not be paid for by a `--no-verify` run.
        from .brokers.alpaca_paper import AlpacaPaperBroker

        client = AlpacaPaperBroker(api_key=api_key, secret_key=secret_key)
        account = client.get_account()
        quote = client.get_quote("SPY")
        print(f"✓ Compte {account.account_id} · cash {account.cash:,.2f} {account.currency}")
        print(f"✓ SPY bid {quote.bid:.2f} / ask {quote.ask:.2f}")
        print("\nLes clés fonctionnent. Tu peux lancer :")
        print("  tradingbench smoke --broker alpaca --place-order")
        return 0
    except Exception as exc:  # noqa: BLE001 - on veut le message, pas la stack
        print(f"✗ Échec : {exc}")
        print("  Les clés sont enregistrées mais inutilisables telles quelles.")
        print("  Vérifie qu'elles viennent bien de l'onglet *Paper Trading* (pas du compte live).")
        return 1
