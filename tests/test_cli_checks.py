"""The smoke test must fail on empty payloads, not just on exceptions."""

from __future__ import annotations

from tradingbench.cli import _run


def test_empty_payload_is_a_failure():
    check = _run(
        "bars",
        lambda: [],
        lambda bars: f"{len(bars)} barres",
        validate=lambda bars: len(bars) >= 2,
    )
    assert check.ok is False
    assert "réponse inattendue" in check.detail


def test_valid_payload_passes():
    check = _run(
        "bars", lambda: [1, 2, 3], lambda bars: f"{len(bars)} barres", validate=len
    )
    assert check.ok is True
    assert "3 barres" in check.detail


def test_exception_is_reported_not_raised():
    def boom():
        raise RuntimeError("backend down")

    check = _run("account", boom, str)
    assert check.ok is False
    assert "RuntimeError" in check.detail
    assert "backend down" in check.detail


def test_check_without_validator_defaults_to_success():
    assert _run("clock", lambda: True, str).ok is True
