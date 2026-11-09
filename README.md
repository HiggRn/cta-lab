# CTA Lab

Multi-asset CTA trend-following backtest system.

## Layout

- `common.py`: core interfaces: `Bar`, `Indicator`, `HistoryContext`, `AlphaModel`, `RiskModel`, `PortfolioConstructionModel` (PCM), `Strategy`
- `indicators/`: reusable indicators
- `panel.py`: `Panel`, loads raw data + asset map into the daily timeline
- `engine.py`: `run_backtest()`, runs a `Strategy` over a `Panel`
- `baselines.py`: the course baselines and benchmarks (`BASELINE_STRATEGIES`) and their models
- `strategy.py`: the team's strategies (`USER_STRATEGIES`): the daily TSMOM example, asset-class risk parity from the midterm report, the daily TSMOM at 11% vol (our best result after costs), the EWMA vol target with a vol floor, and a 50/50 mix of the two
- `main.py`: the entry point. Wires data paths, cost model and params, runs every baseline and user strategy and calls the analysis
- `oos/`: out-of-sample code, parked for now (see `oos/README.md`). Not used by `main.py` or the tests in `tests/`
- `analysis.py`: performance statistics, regressions, benchmark and SG Trend comparison, risk model check, lookback study, asset-class and autocorrelation statistics
- `tests/`: `uv run pytest`
- `eda.py`, `benchmark_comparison.py`: legacy scripts, kept for reference

## Data

- `data/raw/*.csv`: in-sample daily prices, to Dec-2014
- `data/oos/*.csv`: out-of-sample prices (2015+ rows only, a public-data extension to 2024-03, notes in `data/oos_meta/`). Used only by `oos/run_oos.py`
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
| Risk | `SampleCovarianceRiskModel` | covariance of the last `risk_lookback` daily returns |
| Risk | `ExpandingMonthlyRiskModel` (baselines) | covariance of all past monthly returns, at least 36 months. Use `risk_lookback=None` |
| Risk | `EwmaMonthlyRiskModel` (baselines) | EWMA covariance of monthly returns (12-month half-life), reacts faster |
| Risk | `VolFloorRiskModel` (strategy) | wraps a risk model and raises any vol below 3%/yr to 3%, keeping correlations |
| Risk | `EwmaDailyRiskModel` (baselines) | Moskowitz, Ooi and Pedersen (2012): EWMA of daily returns, centre of mass 60 days, diagonal |
| PCM | `InverseVolPCM` | weight = signal / vol, scaled so gross exposure equals `target_gross_leverage` |
| PCM | `EqualRiskPCM` | signal x 40% / vol, then 1/N; flat until 10 symbols are live |
| PCM | `VolTargetPCM` | wraps another PCM and scales its weights so sqrt(w' cov w) equals `target_vol` |
| PCM | `TradeBufferPCM` (strategy) | wraps a PCM: no-trade band of +-20% of each target; trades only to the band's edge |
| PCM | `AssetClassBudgetPCM` (strategy) | wraps a PCM: equal stand-alone risk per asset class (COMM, EQ, FI, FX) |
| Strategy | `SleeveMixStrategy` (strategy) | runs several strategies side by side and holds a fixed-weight mix of their targets |


## Results

Sharpe ratios of monthly returns, before (gross) and after (net) costs, Jun-1976 to Dec-2014, with `FuturesCostModel` and `exec_lag=2` as in `main.py`. Max drawdown is on compounded net wealth. Calmar is the annualised average net return divided by the absolute max drawdown, over the full sample. Costs are annual averages.

| Strategy | Gross SR | Net SR | Max DD | Calmar | Costs (bps/yr) | of which financing |
|---|---|---|---|---|---|---|
| TSMOM 12m (baseline) | 0.92 | 0.78 | -29% | 0.30 | 148 | 115 |
| Static benchmark | 0.46 | 0.37 | -69% | 0.08 | 138 | 115 |
| Vol target 11% | 1.02 | 0.87 | -23% | 0.42 | 170 | 130 |
| Vol target 11%, EWMA risk | 1.15 | 0.95 | -17% | 0.64 | 229 | 179 |
| Blend 1/3/6/12m | 0.81 | 0.69 | -31% | 0.25 | 129 | 82 |
| Blend + EWMA vol target | 1.07 | 0.88 | -24% | 0.43 | 217 | 145 |
| Daily EWMA sizing (MOP 2012) | 1.18 | 0.93 | -23% | 0.50 | 307 | 244 |
| Daily 252d TSMOM, inverse vol | 1.10 | 1.03 | -7% | 0.51 | 27 | 0 |
| Asset-class risk parity (EWMA) | 1.07 | 0.83 | -18% | 0.53 | 282 | 213 |
| **Daily TSMOM 11% vol (floor, buffer)** | **1.38** | **1.20** | **-17%** | **0.85** | **229** | **137** |
| EWMA vol target + floor | 1.14 | 1.01 | -16% | 0.69 | 143 | 108 |
| Mix 50/50: EWMA + floor and daily 11% vol | 1.32 | 1.16 | -15% | 0.83 | 178 | 113 |

- The strategies are the ones discussed in the midterm report (`Summary_Report_Team_1.pdf`, Section 10), the daily TSMOM example, and three extensions that do best after costs, each with one simple reason. The report's continuous signal was dropped there and is not included.
- **Daily TSMOM 11% vol (floor, buffer)** is the best result after costs: net Sharpe 1.20 against 0.95 for the best course baseline ("Vol target 11%, EWMA risk", p = 0.006). It is the daily example scaled to 11% ex-ante vol, with a 3% vol floor (stops leverage piling into near-zero-vol contracts) and a 20% trade buffer (roughly halves daily turnover). Both parameters were fixed before testing.
  - It is the best strategy after costs in every sub-period: net Sharpe 1.45 (1976-89), 1.13 (1990-99), 1.12 (2000-07) and 0.86 (2008-14), where the baseline drops to 0.20.
  - With every cost doubled it still has a net Sharpe of 1.01, against 0.74 for the best course baseline.
  - Each layer earns its place: without the floor, net Sharpe is 1.03 (with the buffer only); without the buffer, 1.17 at 1.3x the costs. Leverage caps added nothing on top and were dropped.
- **EWMA vol target + floor** is the report's best risk model (EWMA covariance) with the same 3% vol floor. The floor cuts its costs from 229 to 143 bps/yr and lifts net Sharpe from 0.95 to 1.01 (p < 0.001), and from 0.74 to 0.88 with costs doubled.
- **Mix 50/50** holds half of each: monthly and daily trend, rebalanced at different speeds, much as CTAs blend horizons (report Section 6). Net Sharpe is 1.16 with the smallest drawdown of the three (-15%). The two sleeves correlate 0.86, so the gain over the daily strategy alone is in drawdown and in relying less on daily trading (turnover 30 against 46 a year), not in Sharpe.
- Variants tested and left out: HRP (complex, skew down to -2.4), risk caps (+0.02 at most), leverage caps (nothing on top of the floor) and the continuous signal (lower Sharpe).
- Financing (50 bps/yr on gross leverage above 1x) is 60-80% of costs for every leveraged strategy, so the gap between gross and net Sharpe mostly tracks gross leverage.
- Asset-class risk parity gives fixed income a full quarter of the risk. Bonds have low volatility, so that needs a lot of notional, and its costs are the second highest (282 bps/yr). Net of costs it trails plain EWMA vol targeting (0.83 against 0.95).
- Trading one day sooner (`exec_lag=1`) adds only 0.02-0.05 in-sample.

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
- This trades about one day later than the lecture convention (trading at the same month-end close as the signal). That convention needs to know a day is the month's last before the next day arrives. Waiting costs the baseline about 0.09 of Sharpe before costs (1.01 to 0.92 with `exec_lag=1`), and `exec_lag=2` another 0.02. The midterm report's 0.99 uses the lecture convention.

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
