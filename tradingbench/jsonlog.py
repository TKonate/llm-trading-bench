"""Structured JSONL logging.

One line per event, append-only. This is the raw material for the dashboard and
for the post-mortem analysis: an agent's decision is worthless if we cannot
replay the exact context and tool calls that produced it. Append-only also
means a crashed run leaves every event written before the crash, which is
exactly the part worth reading.

Deliberately not a queue, a batcher or a rotating file: a lost line is a lost
measurement, and rotation would break the "one experiment, one file" rule.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    # Milliseconds are enough to order events within a session, and the
    # timezone marker keeps logs from different hosts comparable.
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class JsonlLogger:
    """Append-only event sink. One instance per log file, shared by a run."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        """Append one event.

        `event` is the discriminator every consumer switches on, so it is the
        one field that cannot be overridden by a caller's `fields`. The file is
        opened per call rather than held open: the writer may be a cron job that
        exits, and a handle cached across runs would be reopened anyway.
        """
        record = {"ts": _now(), "event": event, **fields}
        # `default=str` keeps an exotic value (Path, datetime, Decimal) from
        # turning a measurement into a TypeError at write time.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Every event in file order; empty list when the file was never written."""
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
