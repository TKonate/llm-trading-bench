"""Saisie des clés API sans passer par un éditeur ni par un chat.

Usage (depuis l'hôte) :
    docker exec -it <conteneur> /opt/data/projects/llm-trading-bench/.venv/bin/tradingbench setkeys

Les valeurs sont lues avec `getpass` (aucun écho), écrites dans .env en
préservant les commentaires, puis le fichier est passé en 600. Un test de
connexion est lancé juste après pour vérifier que les clés fonctionnent
vraiment — une clé qui « a l'air bonne » n'est pas une clé qui marche.
"""

from __future__ import annotations

import getpass
import os
import stat
from pathlib import Path

from dotenv import set_key, unset_key

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def mask(value: str) -> str:
    if not value:
        return "(vide)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…{value[-4:]} ({len(value)} car.)"


def update_env_file(path: Path, values: dict[str, str]) -> None:
    """Écrit des paires clé/valeur dans un fichier .env sans perdre les commentaires."""
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
    """Si on tourne en root (docker exec), aligner le propriétaire sur celui du projet.

    Sans ça, .env appartient à root:root en 600 et les outils qui tournent sous
    l'utilisateur du projet (smoke test, boucle de décision) ne peuvent plus le lire.
    """
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    info = path.parent.stat()
    os.chown(path, info.st_uid, info.st_gid)


def _prompt_secret(label: str) -> str:
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
