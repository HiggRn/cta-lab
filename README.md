# CTA Lab

Multi-asset CTA trend-following backtest system.

## Layout

- `common.py`: core interfaces: `Bar`, `HistoryContext`, `AlphaModel`, `RiskModel`, `PortfolioConstructionModel` (PCM), `Strategy`
- `panel.py`: `Panel`, loads raw data + asset map into the daily timeline
- `engine.py`: `run_backtest()`, runs a `Strategy` over a `Panel`
- `strategy.py`: `ModularStrategy` and the concrete models
- `main.py`: entry point, wires data paths, models and params, then runs the backtest

## Data

- `data/raw/*.csv`
- `data/processed/AssetMapCsv.csv`

## Running

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.
```
uv sync
uv run main.py
```

## Scope

Strategy development should only ever touch:

- `strategy.py`: new models
- `main.py`: which models/data/params to run

`common.py` / `panel.py` / `engine.py` are shared core. If a strategy seems to need a change there, that's worth a separate pull request, not a quiet edit.

## How a strategy works

`ModularStrategy` chains three models:

1. `AlphaModel.update(bars, context)` returns signals. Runs on every bar.
2. `RiskModel.covariance(returns)` returns the annualized covariance of the last `risk_lookback` returns. Runs on rebalance days only.
3. `PortfolioConstructionModel.create_targets(signals, cov)` returns target weights. Runs on rebalance days only.

`ModularStrategy` decides when to rebalance (`rebalance="daily"` or `"monthly"`). The models never do.

Current models: `TsmomAlpha`, `SampleCovarianceRiskModel`, `InverseVolPCM` (weight = signal / vol, scaled so gross exposure equals `target_gross_leverage`).

## Writing an alpha

- `bars` only contains the symbols that traded today.
- `update()` is called on every bar, so stateful alphas stay current.
- Return only the symbols with a new signal. Symbols you leave out keep their last signal. Return `0.0` to go flat.
- `context.get_returns(n)` gives the last `n` daily returns up to today. NaN means the symbol had no bar that day, and the return over a gap lands on the first day back. pandas reductions (`sum`, `prod`, `std`) skip NaN; plain numpy does not.
- Don't shift anything for execution lag.

## Monthly strategy

Use `rebalance="monthly"`. Targets are built on the panel-wide month end (`is_calendar_month_end`).

- A symbol with no bar on that day still gets a weight, from its last signal.
- Don't use `Bar.is_month_end` as the portfolio trigger. It is per-symbol and marks that symbol's own last bar of the month. Inside an alpha it is fine, e.g. to compute a signal only on a symbol's own month end. The signal is carried forward.

## Daily strategy

Use `rebalance="daily"`. Same alpha rules. Symbols with no bar today keep their last signal and weight, so there is nothing to merge by hand.

## Execution and costs

- Targets from day `t` are traded `exec_lag` bars later. With the default `exec_lag=1` they first earn day `t+1`'s return.
- Between rebalances positions are held, not rebalanced. Weights drift with prices.
- Cost = `cost_bps` × traded weight (`|target - drifted weight|`) on each rebalance.
- A held symbol with no bar earns 0 that day. The first day back earns the whole gap.

## Other notes

- `ModularStrategy` keeps state (last signals). Create a new instance for every backtest.
- Symbols with too little history (NaN vol) get no weight.
- No margin modeling: positions sized like equities, shorting allowed
- No rollover modeling: assume data has been rolled over already
- Keep it simple: add a new model class only when a real strategy needs one
- Lookahead is designed to be structurally impossible, not just documented as forbidden
