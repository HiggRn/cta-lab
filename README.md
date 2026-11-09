# CTA Lab

Multi-asset CTA trend-following backtest system.

## Layout

- `common.py`: core interfaces: `Bar`, `Indicator`, `HistoryContext`, `AlphaModel`, `RiskModel`, `PortfolioConstructionModel` (PCM), `Strategy`
- `indicators/`: reusable indicators
- `panel.py`: `Panel`, loads raw data + asset map into the daily timeline
- `engine.py`: `run_backtest()`, runs a `Strategy` over a `Panel`
- `baselines.py`: the course baselines and benchmarks (`BASELINE_STRATEGIES`) and their models
- `strategy.py`: the team's strategies (`USER_STRATEGIES`) and the enhancement models (vol floor, leverage and risk caps, trade buffer, HRP, continuous signal, strategy mix)
- `main.py`: the entry point. Wires data paths, cost model and params, runs every baseline and user strategy and calls the analysis
- `oos.py`: `check_splice()`, checks out-of-sample price files join smoothly onto the in-sample ones
- `analysis.py`: performance statistics, regressions, benchmark and SG Trend comparison, risk model check, lookback study, asset-class and autocorrelation statistics
- `tests/`: `uv run pytest`
- `eda.py`, `benchmark_comparison.py`: legacy scripts, kept for reference

## Data

- `data/raw/*.csv`: in-sample daily prices, to Dec-2014
- `data/oos/*.csv`: out-of-sample prices (2015+ rows only, currently a public-data extension to 2024-03, notes in `data/oos_meta/`), same format as `data/raw`. When filled, `main.py` checks each file continues smoothly from its last in-sample close (`oos.check_splice`, saved to `outputs/oos_splice_check.csv`), runs every backtest straight through the join, and adds out-of-sample tables from 2015-01
- `data/processed/AssetMapCsv.csv`
- `data/processed/MonthlyReturns.csv` (course file, used to check our monthly returns)
- `data/processed/Lecture3_livedata.xlsx` (course file with the SG Trend index)
- `data/external/`: Fama-French factors, downloaded automatically from the Ken French library on first use

## Running

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.
```
uv sync
uv run main.py      # several minutes with all strategies; tables are also saved to outputs/
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

The `schedule` passed to `ModularStrategy` decides when to rebalance (`DailySchedule`, `WeeklySchedule` or `FirstAllTradableDayMonthlySchedule`). The models never do.

Current models (`common.py` unless noted):

| Kind | Model | What it does |
|---|---|---|
| Alpha | `MonthlyTsmomAlpha` (baselines) | sign of the summed completed monthly returns. Several lookbacks are averaged, e.g. `(1, 3, 6, 12)` |
| Alpha | `StaticLongAlpha` (baselines) | static benchmark: always long, with the same history requirement as TSMOM |
| Alpha | `DailyTsmomAlpha` (strategy) | sign of the return over the last 252 daily bars |
| Alpha | `ContinuousTsmomAlpha` (strategy) | trend strength instead of sign: t-stat of the last 12 monthly returns, capped at +-1 |
| Risk | `SampleCovarianceRiskModel` | covariance of the last `risk_lookback` daily returns |
| Risk | `ExpandingMonthlyRiskModel` (baselines) | covariance of all past monthly returns, at least 36 months. Use `risk_lookback=None` |
| Risk | `EwmaMonthlyRiskModel` (baselines) | EWMA covariance of monthly returns (12-month half-life), reacts faster |
| Risk | `EwmaDailyRiskModel` (baselines) | Moskowitz, Ooi and Pedersen (2012): EWMA of daily returns, centre of mass 60 days, diagonal |
| Risk | `VolFloorRiskModel` (strategy) | wraps a risk model and raises any vol below 3%/yr to 3%, keeping correlations |
| PCM | `InverseVolPCM` | weight = signal / vol, scaled so gross exposure equals `target_gross_leverage` |
| PCM | `EqualRiskPCM` | signal x 40% / vol, then 1/N; flat until 10 symbols are live |
| PCM | `VolTargetPCM` | wraps another PCM and scales its weights so sqrt(w' cov w) equals `target_vol` |
| PCM | `LeverageCapPCM` (strategy) | wraps a PCM: at most 1x capital per contract, at most 5x gross |
| PCM | `RiskCapPCM` (strategy) | wraps a PCM: no contract above 10% of portfolio variance (Euler contribution); total vol kept |
| PCM | `AssetClassBudgetPCM` (strategy) | wraps a PCM: equal stand-alone risk per asset class |
| PCM | `TradeBufferPCM` (strategy) | wraps a PCM: no-trade band of +-20% of each target; trades only to the band's edge |
| PCM | `HrpPCM` (strategy) | Hierarchical Risk Parity (Lopez de Prado 2016) on signed positions |
| Strategy | `SleeveMixStrategy` (strategy) | runs several strategies side by side and holds a fixed-weight mix of their targets |

The enhancement parameters (3% vol floor, 1x / 5x leverage caps, 10% risk cap, 20% buffer, 50/50 mix) were fixed before testing, not tuned.

## Results

Sharpe ratios of monthly returns, before (gross) and after (net) costs, with `FuturesCostModel` and `exec_lag=2` as in `main.py`. In-sample is Jun-1976 to Dec-2014; out-of-sample is Jan-2015 to Mar-2024 on the public-data extension in `data/oos`.

| Strategy | In-sample gross | In-sample net | Out-of-sample gross | Out-of-sample net |
|---|---|---|---|---|
| TSMOM 12m (baseline) | 0.92 | 0.78 | 0.24 | 0.02 |
| Static benchmark | 0.47 | 0.37 | 0.30 | 0.15 |
| Vol target 11% | 1.02 | 0.87 | 0.25 | 0.01 |
| Vol target 11%, EWMA risk | 1.15 | 0.95 | 0.45 | -0.03 |
| Blend 1/3/6/12m | 0.81 | 0.69 | -0.09 | -0.30 |
| Blend + EWMA vol target | 1.07 | 0.88 | 0.16 | -0.36 |
| Daily EWMA sizing (MOP 2012) | 1.18 | 0.93 | 0.46 | -0.17 |
| Daily 252d TSMOM, inverse vol | 1.10 | 1.03 | 0.41 | 0.24 |
| TSMOM + vol floor & caps | 0.91 | 0.80 | 0.20 | 0.04 |
| TSMOM + trade buffer | 0.88 | 0.76 | 0.24 | 0.03 |
| EWMA vol target + floor & caps | 1.13 | 1.00 | 0.34 | 0.16 |
| EWMA F&C + risk caps | 1.15 | 1.02 | 0.34 | 0.16 |
| EWMA F&C + risk caps + class budgets | 1.08 | 0.93 | 0.35 | 0.14 |
| EWMA F&C + continuous signal + buffer | 1.06 | 0.96 | 0.36 | 0.22 |
| HRP TSMOM (floor, caps, 11% vol) | 1.04 | 0.85 | 0.48 | 0.17 |
| HRP TSMOM + trade buffer | 0.98 | 0.80 | 0.47 | 0.19 |
| HRP + buffer + risk caps | 1.23 | 1.04 | 0.49 | 0.24 |
| Daily TSMOM 11% vol (floor, caps, buffer) | 1.36 | 1.18 | 0.51 | 0.28 |
| **Mix: EWMA F&C + daily TSMOM 50/50** | **1.30** | **1.15** | **0.43** | **0.24** |

- The 50/50 mix of monthly and daily trend is the best-supported improvement: in-sample net Sharpe 1.15 against 1.00 for "EWMA vol target + floor & caps" (p = 0.002), with the smallest drawdown (-15%).
- The vol floor and leverage caps matter most for strategies that lever up quiet contracts: they cut costs on the EWMA strategy from 229 to 142 bps/yr. The risk cap fixes HRP's concentration (worst month -30% to -10.5%).
- Asset-class budgets did not help. The continuous signal has the lowest costs (100 bps/yr) but a lower gross Sharpe.
- Trading one day sooner (`exec_lag=1`) adds only 0.02-0.05 in-sample.
- Out-of-sample Sharpe ratios have a standard error of about 0.33, so differences between the better strategies there are noise. Trend following was weak for 2015-2024 generally.

## Writing an alpha

- `bars` only contains the symbols that traded today.
- `update()` is called on every bar. Keep one indicator per symbol and feed it every bar you receive.
- An `Indicator` is updated only on days its symbol traded, so "last N bars" means that symbol's own N trading days. Read `.value` only when `.is_ready`.
- Return only the symbols with a new signal. Symbols you leave out keep their last signal. Return `0.0` to go flat.
- `context.get_returns(n)` gives the last `n` daily returns up to today across all symbols. NaN means the symbol had no bar that day, and the return over a gap lands on the first day back. pandas reductions (`sum`, `prod`, `std`) skip NaN; plain numpy does not.
- Don't shift anything for execution lag.

## Monthly strategy

Use `FirstAllTradableDayMonthlySchedule`. A month's end is only known once the next month's first bar arrives, so the strategy rebalances on the first day of the new month on which every active symbol has traded (waiting at most 5 days), using completed months only.

- A symbol with no bar on that day still gets a weight, from its last signal.
- This trades about one day later than the lecture convention (trading at the same month-end close as the signal), which costs the baseline about 0.07 of Sharpe before costs.

## Daily strategy

Use `DailySchedule`. Same alpha rules. Symbols with no bar today keep their last signal and weight, so there is nothing to merge by hand.

## Execution and costs

- Targets from day `t` are traded `exec_lag` bars later. With `exec_lag=1` they first earn day `t+1`'s return; `main.py` uses `exec_lag=2`.
- Between rebalances positions are held, not rebalanced. Weights drift with prices.
- `FuturesCostModel` (Lecture 9): 1.5 bps per unit of traded weight, roll cost of 1.5 bps x 4 rolls/yr on gross exposure, and financing of 50 bps/yr on gross leverage above 1x. Financing is the largest cost for leveraged strategies.
- A held symbol with no bar earns 0 that day. The first day back earns the whole gap.

## Other notes

- `ModularStrategy` and any alpha holding indicators keep state. Create new instances for every backtest.
- Symbols with too little history (NaN vol) get no weight.
- No margin modeling: positions sized like equities, shorting allowed
- No rollover modeling: assume data has been rolled over already
- Keep it simple: add a new model class only when a real strategy needs one
- Lookahead is designed to be structurally impossible, not just documented as forbidden
