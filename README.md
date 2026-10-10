# CTA Lab

Multi-asset CTA trend-following backtest system.

## Layout

- `common.py`: core interfaces: `Bar`, `Indicator`, `HistoryContext`, `AlphaModel`, `RiskModel`, `PortfolioConstructionModel` (PCM), `Strategy`
- `indicators/`: reusable indicators
- `panel.py`: `Panel`, loads raw data + asset map into the daily timeline
- `engine.py`: `run_backtest()`, runs a `Strategy` over a `Panel`
- `strategy.py`: `ModularStrategy` and the concrete models
- `main.py`: the entry point. Wires data paths, models and params (`STRATEGIES` names every configuration), runs the backtests and calls the analysis
- `analysis.py`: performance statistics, regressions, benchmark and SG Trend comparison, risk model check, lookback study, asset-class and autocorrelation statistics
- `tests/`: `uv run pytest`
- `eda.py`, `benchmark_comparison.py`: legacy scripts, kept for reference

## Data

- `data/raw/*.csv`
- `data/processed/AssetMapCsv.csv`
- `data/processed/MonthlyReturns.csv` (course file, used to check our monthly returns)
- `data/processed/Lecture3_livedata.xlsx` (course file with the SG Trend index)
- `data/external/`: Fama-French factors, downloaded automatically from the Ken French library on first use

## Running

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.
```
uv sync
uv run main.py      # about 80 seconds; tables are also saved to outputs/
uv run pytest
```

## Scope

Strategy development should only ever touch:

- `strategy.py`: new models
- `main.py`: which models/data/params to run
- `indicators/`: any new indicator a model needs

`common.py` / `panel.py` / `engine.py` are shared core. If a strategy seems to need a change there, that's worth a separate pull request, not a quiet edit.

## How a strategy works

`ModularStrategy` chains three models:

1. `AlphaModel.update(bars, context)` returns signals. Runs on every bar.
2. `RiskModel.covariance(returns)` returns the annualized covariance of the last `risk_lookback` returns. Runs on rebalance days only.
3. `PortfolioConstructionModel.create_targets(signals, cov)` returns target weights. Runs on rebalance days only.

`ModularStrategy` decides when to rebalance (`rebalance="daily"` or `"monthly"`). The models never do.

Current models:

| Kind | Model | What it does |
|---|---|---|
| Alpha | `TsmomAlpha` | sign of the return over the last `lookback` daily bars |
| Alpha | `MonthlyTsmomAlpha` | sign of the summed past monthly returns, on each symbol's own month end. Several lookbacks are averaged, e.g. `(1, 3, 6, 12)` |
| Alpha | `StaticLongAlpha` | static benchmark: always long, with the same history requirement as TSMOM |
| Risk | `SampleCovarianceRiskModel` | covariance of the last `risk_lookback` daily returns |
| Risk | `ExpandingMonthlyRiskModel` | covariance of all past monthly returns, at least 36 months. Use `risk_lookback=ALL_HISTORY` |
| Risk | `EwmaMonthlyRiskModel` | EWMA covariance of monthly returns (12-month half-life), reacts faster |
| Risk | `EwmaDailyRiskModel` | Moskowitz, Ooi and Pedersen (2012): EWMA of daily returns, centre of mass 60 days, diagonal |
| PCM | `InverseVolPCM` | weight = signal / vol, scaled so gross exposure equals `target_gross_leverage` |
| PCM | `EqualRiskPCM` | signal x 40% / vol, then 1/N; flat until 10 symbols are live |
| PCM | `VolTargetPCM` | wraps another PCM and scales its weights so sqrt(w' cov w) equals `target_vol` |

## Results

`STRATEGIES["TSMOM 12m (baseline)"]` is the monthly 12-month TSMOM baseline. Monthly returns from Feb-1976 to Dec-2014, no costs:

| Strategy | Avg | Std | Sharpe | Max DD |
|---|---|---|---|---|
| TSMOM 12m (baseline) | 11.1% | 10.9% | 1.01 | -25% |
| Static benchmark | 6.8% | 14.4% | 0.47 | -68% |
| Vol target 11% (expanding covariance) | 11.9% | 10.9% | 1.10 | -19% |
| Vol target 11%, EWMA risk | 13.6% | 11.1% | 1.23 | -14% |
| Blend 1/3/6/12m | 9.3% | 9.4% | 0.99 | -13% |
| Daily EWMA sizing (MOP 2012) | 15.6% | 12.1% | 1.29 | -15% |

These are in-sample and before costs, so treat the variants as ideas to test, not results.

## Writing an alpha

- `bars` only contains the symbols that traded today.
- `update()` is called on every bar. Keep one indicator per symbol and feed it every bar you receive.
- An `Indicator` is updated only on days its symbol traded, so "last N bars" means that symbol's own N trading days. Read `.value` only when `.is_ready`.
- Return only the symbols with a new signal. Symbols you leave out keep their last signal. Return `0.0` to go flat.
- `context.get_returns(n)` gives the last `n` daily returns up to today across all symbols. NaN means the symbol had no bar that day, and the return over a gap lands on the first day back. pandas reductions (`sum`, `prod`, `std`) skip NaN; plain numpy does not.
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

- `ModularStrategy` and any alpha holding indicators keep state. Create new instances for every backtest.
- Symbols with too little history (NaN vol) get no weight.
- No margin modeling: positions sized like equities, shorting allowed
- No rollover modeling: assume data has been rolled over already
- Keep it simple: add a new model class only when a real strategy needs one
- Lookahead is designed to be structurally impossible, not just documented as forbidden
