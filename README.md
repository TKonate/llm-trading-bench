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
| 1 — Broker abstraction + Alpaca paper client + smoke test | ✅ done (validated on a real paper account) |
| 2 — Model tool-calling harness (eliminatory) | ✅ done (deepseek-v4.1-flash: 6/6, GO) |
| 3 — Decision loop + self-critique + JSONL logs | ✅ done (scheduled, alerts on failure only) |
| 4 — Dashboard + QQQ & random-portfolio benchmark | ⬜ next |

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

# reuse an existing Hermes/OpenAI-compatible credential without retyping it
.venv/bin/tradingbench importkeys --provider opencode-go

# eliminatory tool-calling harness: a model that cannot call its tools does not compete
.venv/bin/tradingbench tooleval --model deepseek-v4.1-flash

# one decision session (slot detected from the clock; --force to ignore it)
.venv/bin/tradingbench session --broker alpaca
.venv/bin/tradingbench session --broker alpaca --dry-run   # real data, simulated orders

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

**Capital is enforced in code, not by the broker.** Alpaca has no reset-balance endpoint (404),
so the agent's budget (1 000 € ≈ 1 080 $) is a hard notional cap inside the order executor: any buy
that would push total exposure past the budget is refused. Fractional quantities are allowed —
with a 1 000 € budget and a 25 % position cap, a single SPY share at ~760 $ would not fit otherwise.

**Every buy can carry a stop-loss and a take-profit.** Both map to an Alpaca bracket order (a single
one maps to OTO). The executor refuses a protection placed on the wrong side of the price: a stop
above the market is not a stop, it is a disguised immediate sale.

**News are data, not instructions.** Headlines come from Alpaca News (Benzinga) and from an optional
Markdown brief written by another agent (Palantir). Both are injected with an explicit "untrusted
external content" warning in the prompt, because a headline is attacker-controlled text.

**The tool harness is multi-turn and executes the tools.** Judging a model on its first message would
score myopia, not competence: a good agent checks the price *then* orders. Each scenario therefore runs
a capped tool loop (4 turns) against a mock broker, with its own portfolio — including seeded positions
for sell scenarios. Scoring is strict: a wrong symbol, a wrong quantity or a forbidden order is a
failure, and the model only joins the benchmark at 100 %.

## Layout

```
tradingbench/
  brokers/    base.py (contract) · mock.py · alpaca_paper.py · __init__.py (factory)
  llm/        client.py (OpenAI-compatible, tool calling) · providers.py
  harness/    scenarios.py · executor.py · runner.py
  models.py   Account, Position, Quote, Bar, Order
  config.py   env-driven settings
  jsonlog.py  append-only JSONL logger
  cli.py      doctor · smoke · setkeys · importkeys · tooleval
tests/        61 tests: broker contract, Alpaca mapping, tool-calling scoring
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
