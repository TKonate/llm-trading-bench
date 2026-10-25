# llm-trading-bench

Multi-agent LLM paper-trading benchmark. Several LLM agents get the **same tools, the same data
and the same rules**, trade a virtual portfolio, and we measure honestly whether any of them has an
edge — against the market **and** against random portfolios.

> Educational project. No financial advice. No real money until the system clears a documented
> go-live gate (see the Obsidian project note). Default assumption: it will *not* beat the market,
> and that is a valid result.

## Status

| Livrable | État |
|---|---|
| 1 — Broker abstraction + Alpaca paper client + smoke test | ✅ done |
| 2 — Model tool-calling harness (eliminatory) | ⬜ next |
| 3 — Decision loop + self-critique + JSONL logs | ⬜ |
| 4 — Dashboard + QQQ & random-portfolio benchmark | ⬜ |

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # fill in Alpaca paper keys if you want the live paper API
```

## Use

```bash
# offline, no account, no network — proves the plumbing works
.venv/bin/tradingbench smoke --broker mock

# store Alpaca paper keys in .env without any editor (hidden input, then live check)
.venv/bin/tradingbench setkeys

# against a real (paper) Alpaca account
.venv/bin/tradingbench smoke --broker alpaca
.venv/bin/tradingbench smoke --broker alpaca --place-order   # also tests the order path

# configuration check, no network call
.venv/bin/tradingbench doctor

# tests
.venv/bin/pytest
.venv/bin/ruff check .
```

## Design decisions that matter

**The decision loop never imports Alpaca.** `tradingbench.brokers.base.Broker` is the only surface an
agent-facing tool sees (`get_account`, `get_positions`, `get_quote`, `get_bars`, `place_order`,
`get_order`, `cancel_order`, `is_market_open`). Adding a live broker in phase B means one new
subclass — not a rewrite.

**Two implementations, one contract test.** `MockBroker` (deterministic, offline) and
`AlpacaPaperBroker` (raw REST over httpx, no SDK) both pass the same parametrised test suite. The
Alpaca side is exercised against a fake HTTP transport, so parsing, order payload and error mapping
are really tested without touching the network.

**Raw REST over `alpaca-py`.** Six endpoints do not justify a large dependency tree, and the explicit
mapping is what we will need anyway when the live broker arrives with different fields.

**Logs are JSONL, append-only.** Every tool call and decision gets one line, so an agent's output can
be replayed later. This is the raw material for the dashboard and for post-mortems.

## Layout

```
tradingbench/
  brokers/    base.py (contract) · mock.py · alpaca_paper.py · __init__.py (factory)
  models.py   Account, Position, Quote, Bar, Order
  config.py   env-driven settings
  jsonlog.py  append-only JSONL logger
  cli.py      `tradingbench doctor` / `smoke`
tests/        contract tests (mock + fake Alpaca) and mapping tests
```

## Roadmap

1. **Livrable 2** — scripted tool-calling test (buy / sell / hold / invalid request) run before any
   model joins the experiment. A model that fails is excluded, exactly like Mistral in the video that
   inspired this project (−7 % of pure execution friction over one week).
2. **Livrable 3** — 2 decision sessions per day, market hours only, self-critique every 5 trades,
   hard API budget cap, automatic pause after 3 consecutive tool failures.
3. **Livrable 4** — ranking, equity curves vs QQQ, and the random-portfolio control group that makes
   a +13 % month readable as luck or skill.

## Licence

MIT.
