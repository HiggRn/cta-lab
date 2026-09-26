# CTA Lab

Multi-asset CTA trend-following backtest system.

## Layout

- `common.py`: core interfaces, including `Bar`, `Indicator`, `Strategy`
- `panel.py`: `Panel`, which loads raw data + asset map into the daily timeline
- `engine.py`:`run_backtest()`, which runs a `Strategy` over a `Panel`
- `indicators/`: reusable indicators
- `strategy.py`: strategy implementations
- `main.py`: entry point, which wires up data paths, picks a strategy and runs the backtest

## Data

- `data/raw/*.csv`
- `data/processed/AssetMapCsv.csv`
- `data/processed/MonthlyReturns.csv`

## Running

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.
```
uv sync
uv run main.py
```

## Scope

Strategy development should only ever touch:

- `strategy.py`: the strategy itself
- `main.py`: which strategy/data/params to run
- `indicators/`: any new indicator the strategy needs

`common.py` / `panel.py` / `engine.py` are shared core. If a strategy seems to need a change there, that's worth a separate pull request, not a quiet edit.

## Developing a monthly strategy

`on_bar()` is called once per trading day for the whole panel, even though you may only rebalance monthly:

1. Feed every incoming bar to your per-symbol indicators on every call, as they're stateful and need to stay current, not just get updated on rebalance days.
2. Rebalance on the panel-wide month-end, not on any individual symbol's own `is_month_end`, which is a per-symbol flag (different exchanges, different holiday calendars, won't line up across symbols). Use `is_calendar_month_end` for the portfolio-level trigger instead.
3. On a rebalance day, read `.value` off your indicators and compute target weights.
4. Any other day, return `None`. The engine will hold your last weights for you.
5. Don't hand-roll the execution lag. Whatever weights you return today only start affecting positions `exec_lag` days later.

## Developing a daily strategy

1. `on_bar()` runs every day. Return new weights whenever your signal updates.
2. Not every symbol has a bar every day. If you return a dict missing a symbol, that symbol goes to zero for the day. It does **not** hold the previous weight. If you want a symbol to keep its weight through a data gap, carry your own weights state inside the strategy and merge it in yourself before returning.
3. Same lag rule as above. Don't shift anything yourself.

## Other notes

- No margin modeling: positions sized like equities, shorting allowed
- No rollover modeling: assume data has been rolled over already
- Keep it simple: don't build out a signal/sizing hierarchy until a real strategy actually needs one
- Lookahead is designed to be structurally impossible, not just documented as forbidden
