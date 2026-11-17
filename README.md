# CTA Lab

Multi-asset CTA trend-following backtest system.

## Layout

- `common.py`: core interfaces: `Bar`, `Indicator`, `HistoryContext`, `AlphaModel`, `RiskModel`, `PortfolioConstructionModel` (PCM), `Strategy`
- `indicators/`: reusable indicators (`Momentum`, `MonthlyReturn`, `YangZhangVolatility`, `EwmaVolatility`, `TrendProfile`, `TrendTStat`), imported as `from indicators import ...`
- `config.py`: data paths, the in-sample window, and the contracts that are dropped, chained or roll-fixed
- `panel.py`: `Panel`, loads raw data + asset map into the daily timeline
- `engine.py`: `run_backtest()`, runs a `Strategy` over a `Panel`
- `baselines.py`: the baseline (`Baseline TSMOM`), the midterm's monthly baselines and the static benchmark (`BASELINE_STRATEGIES`), and their models
- `strategy.py`: the team's strategies (`USER_STRATEGIES`): the daily TSMOM example, the same example with Yang-Zhang risk and correlation-adjusted sizing, asset-class risk parity from the midterm report, the daily TSMOM at 11% vol, the EWMA vol target with a vol floor, a 50/50 mix of the two, the trend profile strategies, and our final two, KV1 and KV2
- `main.py`: the entry point. Wires data paths, cost model and params, runs every baseline and user strategy once over the full sample and reports in-sample and out-of-sample
- `analysis.py`: performance statistics, regressions, benchmark and SG Trend comparison, risk model check, lookback study, lag profile, asset-class and autocorrelation statistics
- `tests/`: `uv run pytest`
- `scripts/`: one-off checks, not used by `main.py`: `audit_data.py` (data audit), `eda.py`, and `oos_gap.py` / `oos_lecture.py` (why the plain 12-month TSMOM earns little after 2014)

## Data

- `data/full_raw/*_RAD.CSV`: course daily prices, 1969 to Jun-2025, no header row. `main.py` and the tests read only these
- `data/raw/*.csv`: the earlier in-sample files (to Dec-2014), used only by `scripts/audit_data.py` to check `full_raw` against them
- `data/processed/AssetMapCsv.csv`
- `data/processed/MonthlyReturns.csv` (course file, used to check our monthly returns) and `MonthlyReturns_FullSample.xlsx` (the same to May-2025, used by `scripts/oos_lecture.py`)
- `data/processed/Lecture3_livedata.xlsx` (course file with the SG Trend index)
- `data/external/`: Fama-French factors, downloaded automatically from the Ken French library on first use

`config.py` cleans the universe before anything runs:

- Dropped: duplicates (ES and SC of SP, EN of ND, YM of ZD), contracts with holes or that stop early (BC, BG, ZH, FF, RL; its mini ER is kept) and AP (no data for most of 2022-2024).
- Chained: SP, ND and ZD continue on their mini contracts (ES, EN, YM) once the big contracts stop trading.
- Roll fix: LB jumps from about 500 to 800 on 2023-05-03, when the lumber contract changed. Prices from that day on are rescaled so the jump is not a return.

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
| Alpha | `DailyTsmomAlpha` (baselines) | sign of the return over the last 252 daily bars |
| Alpha | `TrendProfileAlpha` (strategy) | average capped z-score of four pieces of the last 12 months (months 1, 2-3, 4-6, 7-12) minus the z-score of months 13-15. Optionally a constant per asset class |
| Alpha | `LearnedProfileAlpha` (strategy) | the same five pieces, with their weights and a lean per asset class re-estimated every January by least squares on past data only |
| Alpha | `NoShortAlpha` (strategy) | wraps an alpha and sets short signals to 0 in equity index futures |
| Alpha | `TrendStrengthAlpha` (strategy) | t-stat of the 252-bar return (log return over its own daily vol x sqrt(252)), capped at +-1 |
| Risk | `SampleCovarianceRiskModel` | covariance of the last `risk_lookback` daily returns |
| Risk | `ExpandingMonthlyRiskModel` (baselines) | covariance of all past monthly returns, at least 36 months. Use `risk_lookback=None` |
| Risk | `EwmaMonthlyRiskModel` (baselines) | EWMA covariance of monthly returns (12-month half-life), reacts faster |
| Risk | `VolFloorRiskModel` (strategy) | wraps a risk model and raises any vol below 3%/yr to 3%, keeping correlations |
| Risk | `YangZhangRiskModel` (strategy) | sample correlations of the last `risk_lookback` daily returns, with each vol replaced by its Yang-Zhang estimate (open, high, low, close). Needs `update(bars)` on every bar |
| Risk | `EwmaDailyRiskModel` (baselines) | Moskowitz, Ooi and Pedersen (2012): EWMA of daily returns, centre of mass 60 days, diagonal |
| Risk | `EwmaVolRiskModel` (strategy) | sample correlations of the last `risk_lookback` daily returns, with EWMA vols (centre of mass 60 days) on the diagonal |
| PCM | `InverseVolPCM` | weight = signal / vol, scaled so gross exposure equals `target_gross_leverage` |
| PCM | `CorrelationAdjustedPCM` (strategy) | weight = signal / (vol x sum of the symbol's positive correlations, itself included), scaled to `target_gross_leverage`. Contracts that move with many others get less |
| PCM | `EqualRiskPCM` | signal x 40% / vol, then 1/N; flat until 10 symbols are live |
| PCM | `VolTargetPCM` | wraps another PCM and scales its weights so sqrt(w' cov w) equals `target_vol` |
| PCM | `TradeBufferPCM` (strategy) | wraps a PCM: no-trade band of +-20% of each target; trades only to the band's edge |
| PCM | `PositionCapPCM` (strategy) | wraps a PCM: no contract above +-100% of capital |
| PCM | `AssetClassBudgetPCM` (strategy) | wraps a PCM: equal stand-alone risk per asset class (COMM, EQ, FI, FX) |
| PCM | `TranchePCM` (strategy) | wraps a PCM and holds the average of its last four targets, i.e. four tranches of a quarter each |
| PCM | `RiskParitySleevePCM` (strategy) | wraps a PCM: 25% of the risk goes to a long-only book of equity and bond futures, equal risk per class |
| Schedule | `FixedCycleSchedule` (baselines) | rebalances on the first trading day and every 21 trading days after it |
| Schedule | `TrancheSchedule` (strategy) | rebalances on days 0, 5, 10 and 15 of every 21-trading-day cycle, one tranche at a time |
| Strategy | `RollingVolTsmomStrategy` (baselines) | the baseline: sign / 252-day vol x 40% every 21 trading days, the whole book scaled to 11% ex-ante vol with the covariance of all past daily returns. It builds the book itself, because sizing and scaling use different risk estimates |
| Strategy | `SleeveMixStrategy` (strategy) | runs several strategies side by side and holds a fixed-weight mix of their targets |


## Results

One run of every strategy over the full sample (`uv run main.py`): `data/full_raw`, `exec_lag=2`, 2 bps per unit traded plus 2 bps x 4 rolls a year, no financing. In-sample is Jun-1976 (the first month every strategy is live) to Dec-2014, out-of-sample Jan-2015 to Jun-2025. Returns, vol and Sharpe ratios are annualised from monthly returns, and max drawdown is on compounded wealth. Turnover and costs are per year. Leverage is average gross exposure, #Names the average number of positions and Imbalance the average net exposure, in units of capital. `main.py` saves the same numbers to `outputs/strategy_comparison_is.csv` and `outputs/strategy_comparison_oos.csv`.

### Baseline

`Baseline TSMOM` (`RollingVolTsmomStrategy`): every 21 trading days each contract gets the sign of its 252-day return divided by its 252-day vol, times 40%, and the whole book is scaled to 11% ex-ante vol with the covariance of all past daily returns. It waits for 10 contracts. The midterm's monthly version (`TSMOM 12m, monthly`: completed calendar months, expanding monthly vol, 1/N) is kept for comparison.

### Final strategies: KV1 and KV2

KV1 is the baseline with four changes and KV2 is KV1 with two more. Each row adds one change to the row above. Sharpe ratios are after costs; leverage and the largest single position (in units of capital) are in-sample / out-of-sample.

| Step | Change | SR in-sample | SR out-of-sample | Leverage | Largest position |
|---|---|---|---|---|---|
| 0 | Baseline | 1.09 | -0.02 | 4.9 / 7.7 | 6.1 / 6.4 |
| 1 | No shorts in equity index futures | 1.11 | 0.10 | 5.0 / 7.8 | 6.1 / 6.4 |
| 2 | EWMA vol, 12-month correlations | 1.22 | 0.13 | 6.9 / 11.7 | 23.7 / 15.8 |
| 3 | 3% vol floor | 1.25 | 0.09 | 3.8 / 4.3 | 0.8 / 0.5 |
| 4 | Four tranches = **KV1** | **1.28** | **0.14** | 3.6 / 4.0 | 0.8 / 0.5 |
| 5 | Trend strength instead of the sign | 1.27 | 0.12 | 3.3 / 3.6 | 0.9 / 0.5 |
| 6 | Risk-parity sleeve = **KV2** | **1.34** | **0.24** | 3.4 / 3.6 | 1.3 / 0.5 |

1. **No shorts in equity index futures** (`NoShortAlpha`). Stock indices drift up and bounce hard after selloffs, so the short side of the 12-month signal mostly loses there. Equity longs stay, and bonds, commodities and currencies can still go short.
2. **Faster risk estimates** (`EwmaVolRiskModel`). Each contract's vol is an EWMA of daily returns (centre of mass 60 days, as in Moskowitz, Ooi and Pedersen 2012) instead of the 252-day vol, and the 11% target uses correlations from the last 12 months instead of all history. Most of the step is the 12-month window (1.19 / 0.18 on its own); EWMA vol on top adds 0.03 in-sample and costs 0.05 out-of-sample.
3. **3% vol floor** (`VolFloorRiskModel`), the same one as in the daily 11% vol strategy. With faster vol, 1/vol sizing piles into Eurodollar and short sterling: 6.9x gross on average and up to 24x capital in one contract. The floor brings that to 3.8x and under 1x for about the same Sharpe.
4. **Four tranches** (`TrancheSchedule`, `TranchePCM`). The book is split into four parts with a quarter of the capital each, each rebuilt every 21 trading days, a week after the one before. With one book a month the result depends a lot on which day of the month it trades.
5. **Trend strength instead of the sign** (`TrendStrengthAlpha`): the 12-month t-stat capped at +-1, so a weak or noisy trend gets a smaller position. It lowers leverage and turnover but adds nothing to Sharpe, here or with the sleeve (KV1 plus the sleeve with the sign signal makes 1.34 / 0.25).
6. **Risk-parity sleeve** (`RiskParitySleevePCM`). A quarter of the risk goes to a long-only book of every equity and bond future, sized 1/vol with equal risk for the two classes. This part is not trend following: it earns the equity and bond premium, which helps in the years trend does badly.

- Against the baseline the in-sample gain is clear (p = 0.005 for KV1 and 0.006 for KV2), the out-of-sample one is not (p = 0.33 and 0.20). Ten years is short, and 2015-2025 was a bad decade for trend.
- KV2 has the best out-of-sample Sharpe of the strategies in the repo; only the static long benchmark did better (0.36). In-sample the trend profile strategies are well ahead (1.62-1.83).
- With every cost doubled: KV1 1.23 / 0.08, KV2 1.30 / 0.19, baseline 1.03 / -0.12.
- Trading later: at `exec_lag` 1, 5, 10 and 21, KV1 makes 1.31, 1.22, 1.12 and 0.95 in-sample (0.13, 0.12, 0.15, 0.18 out-of-sample), and KV2 1.37, 1.29, 1.19 and 1.01 (0.23, 0.23, 0.24, 0.30).
- Because of the sleeve KV2 is more net long: about 1.5x capital on average, against 0.5-0.9x for KV1. The largest single position is 0.76x capital in KV1 and 1.32x in KV2.
- Both pass `test_no_lookahead`.

By sub-period:

| SR after costs | 1976-89 | 1990-99 | 2000-07 | 2008-14 | 2015-19 | 2020-25 |
|---|---|---|---|---|---|---|
| Baseline TSMOM | 1.14 | 1.32 | 1.34 | 0.56 | -0.22 | 0.08 |
| KV1 | 1.33 | 1.58 | 1.32 | 0.78 | 0.02 | 0.26 |
| KV2 | 1.35 | 1.56 | 1.39 | 0.98 | 0.21 | 0.27 |

### All strategies

Before costs, in-sample (Jun-1976 to Dec-2014):

| Strategy | Ret | Std | SR | Max DD | Turnover | Leverage | #Names | Imbalance |
|---|---|---|---|---|---|---|---|---|
| **Baseline TSMOM** | **13.3%** | **11.5%** | **1.16** | **-19.1%** | **16.9** | **4.92** | **41.9** | **1.49** |
| Static benchmark | 6.7% | 14.4% | 0.46 | -67.8% | 1.7 | 3.27 | 39.8 | 3.27 |
| TSMOM 12m, monthly | 10.2% | 11.2% | 0.91 | -28.7% | 8.7 | 3.26 | 39.8 | 0.90 |
| Blend 1/3/6/12m | 9.1% | 11.2% | 0.81 | -26.0% | 21.1 | 2.60 | 32.9 | 0.72 |
| Vol target 11% | 11.3% | 11.1% | 1.02 | -20.6% | 12.0 | 3.57 | 39.8 | 0.99 |
| Vol target 11%, EWMA risk | 12.9% | 11.2% | 1.15 | -15.1% | 15.7 | 4.52 | 39.8 | 1.25 |
| Daily EWMA sizing (MOP 2012) | 14.4% | 12.2% | 1.18 | -16.5% | 18.2 | 5.82 | 41.9 | 2.15 |
| Blend + EWMA vol target | 12.5% | 11.7% | 1.07 | -23.9% | 32.7 | 3.84 | 32.9 | 1.06 |
| Asset-class risk parity (EWMA) | 12.6% | 11.6% | 1.08 | -13.9% | 25.6 | 5.65 | 39.8 | 1.51 |
| EWMA vol target + floor | 12.4% | 10.9% | 1.14 | -15.1% | 11.1 | 3.12 | 39.8 | 0.76 |
| Daily 252d TSMOM, inverse vol | 4.0% | 3.6% | 1.10 | -7.1% | 14.2 | 1.00 | 42.0 | 0.24 |
| Daily TSMOM 11% vol (floor, buffer) | 18.2% | 12.2% | 1.49 | -18.1% | 49.8 | 3.80 | 42.0 | 0.84 |
| Daily 252d TSMOM, YZ + corr adjusted | 4.3% | 3.4% | 1.26 | -6.7% | 17.8 | 1.00 | 42.0 | 0.23 |
| Mix 50/50: EWMA + floor and daily 11% vol | 15.3% | 10.8% | 1.41 | -13.6% | 32.2 | 3.29 | 42.0 | 0.80 |
| Trend profile | 23.9% | 13.0% | 1.84 | -8.4% | 58.8 | 3.04 | 41.8 | 0.68 |
| Trend profile + equity/bond premium | 24.3% | 12.6% | 1.93 | -9.2% | 51.8 | 3.17 | 41.8 | 1.32 |
| Trend profile, learned weights | 22.5% | 12.9% | 1.75 | -9.8% | 67.8 | 3.17 | 41.8 | 1.66 |
| **KV1** | **15.9%** | **11.9%** | **1.33** | **-19.7%** | **15.3** | **3.57** | **40.4** | **0.93** |
| **KV2** | **17.1%** | **12.4%** | **1.38** | **-21.7%** | **12.7** | **3.36** | **42.0** | **1.52** |

Before costs, out-of-sample (Jan-2015 to Jun-2025):

| Strategy | Ret | Std | SR | Max DD | Turnover | Leverage | #Names | Imbalance |
|---|---|---|---|---|---|---|---|---|
| **Baseline TSMOM** | **0.9%** | **12.0%** | **0.07** | **-35.0%** | **26.7** | **7.72** | **55.4** | **2.36** |
| Static benchmark | 5.7% | 15.0% | 0.38 | -29.2% | 2.0 | 4.50 | 55.4 | 4.50 |
| TSMOM 12m, monthly | -0.1% | 11.1% | -0.01 | -35.0% | 13.8 | 4.50 | 55.4 | 0.66 |
| Blend 1/3/6/12m | -3.9% | 10.6% | -0.37 | -40.0% | 34.1 | 3.56 | 45.5 | 0.58 |
| Vol target 11% | 0.3% | 12.4% | 0.02 | -34.2% | 20.8 | 5.13 | 55.4 | 0.80 |
| Vol target 11%, EWMA risk | 2.0% | 12.2% | 0.16 | -27.3% | 34.4 | 9.01 | 55.4 | 2.62 |
| Daily EWMA sizing (MOP 2012) | 1.9% | 11.4% | 0.17 | -22.1% | 36.1 | 10.19 | 55.4 | 3.85 |
| Blend + EWMA vol target | -1.9% | 11.1% | -0.17 | -31.7% | 76.5 | 7.80 | 45.5 | 2.46 |
| Asset-class risk parity (EWMA) | 2.7% | 12.1% | 0.22 | -28.8% | 56.9 | 11.55 | 55.4 | 2.51 |
| EWMA vol target + floor | 0.8% | 11.4% | 0.07 | -32.3% | 16.7 | 3.92 | 55.4 | 0.46 |
| Daily 252d TSMOM, inverse vol | 0.2% | 2.1% | 0.10 | -3.8% | 13.0 | 1.00 | 55.4 | 0.20 |
| Daily TSMOM 11% vol (floor, buffer) | 1.7% | 12.6% | 0.13 | -24.4% | 55.0 | 4.01 | 55.4 | 0.45 |
| Daily 252d TSMOM, YZ + corr adjusted | 0.5% | 2.4% | 0.23 | -5.7% | 16.5 | 1.00 | 55.4 | 0.24 |
| Mix 50/50: EWMA + floor and daily 11% vol | 1.1% | 11.4% | 0.10 | -27.2% | 37.8 | 3.70 | 55.4 | 0.46 |
| Trend profile | 1.9% | 11.8% | 0.16 | -22.4% | 71.9 | 3.22 | 55.4 | 0.40 |
| Trend profile + equity/bond premium | 2.9% | 11.2% | 0.26 | -20.4% | 62.4 | 3.40 | 55.4 | 1.47 |
| Trend profile, learned weights | 2.5% | 10.9% | 0.23 | -20.1% | 60.8 | 3.62 | 55.4 | 2.54 |
| **KV1** | **2.3%** | **11.8%** | **0.20** | **-28.8%** | **17.5** | **3.98** | **52.8** | **0.55** |
| **KV2** | **3.4%** | **11.5%** | **0.30** | **-23.5%** | **14.7** | **3.62** | **55.4** | **1.45** |

After costs, in-sample:

| Strategy | Ret | Std | SR | Max DD | Turnover | Leverage | #Names | Imbalance | Costs (bps) |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline TSMOM** | **12.6%** | **11.5%** | **1.09** | **-19.3%** | **16.9** | **4.92** | **41.9** | **1.49** | **74** |
| Static benchmark | 6.4% | 14.4% | 0.44 | -68.0% | 1.7 | 3.27 | 39.8 | 3.27 | 30 |
| TSMOM 12m, monthly | 9.8% | 11.2% | 0.87 | -28.9% | 8.7 | 3.26 | 39.8 | 0.90 | 44 |
| Blend 1/3/6/12m | 8.5% | 11.2% | 0.75 | -28.3% | 21.1 | 2.60 | 32.9 | 0.72 | 63 |
| Vol target 11% | 10.7% | 11.0% | 0.97 | -20.8% | 12.0 | 3.57 | 39.8 | 0.99 | 53 |
| Vol target 11%, EWMA risk | 12.2% | 11.2% | 1.09 | -15.4% | 15.7 | 4.52 | 39.8 | 1.25 | 68 |
| Daily EWMA sizing (MOP 2012) | 13.6% | 12.2% | 1.11 | -18.1% | 18.2 | 5.82 | 41.9 | 2.15 | 84 |
| Blend + EWMA vol target | 11.5% | 11.7% | 0.99 | -24.0% | 32.7 | 3.84 | 32.9 | 1.06 | 97 |
| Asset-class risk parity (EWMA) | 11.6% | 11.6% | 1.00 | -15.0% | 25.6 | 5.65 | 39.8 | 1.51 | 97 |
| EWMA vol target + floor | 12.0% | 10.9% | 1.10 | -15.4% | 11.1 | 3.12 | 39.8 | 0.76 | 48 |
| Daily 252d TSMOM, inverse vol | 3.6% | 3.6% | 1.00 | -7.2% | 14.2 | 1.00 | 42.0 | 0.24 | 37 |
| Daily TSMOM 11% vol (floor, buffer) | 16.9% | 12.2% | 1.38 | -18.3% | 49.8 | 3.80 | 42.0 | 0.84 | 131 |
| Daily 252d TSMOM, YZ + corr adjusted | 3.8% | 3.4% | 1.13 | -6.8% | 17.8 | 1.00 | 42.0 | 0.23 | 44 |
| Mix 50/50: EWMA + floor and daily 11% vol | 14.4% | 10.8% | 1.33 | -13.8% | 32.2 | 3.29 | 42.0 | 0.80 | 91 |
| Trend profile | 22.5% | 13.0% | 1.73 | -9.2% | 58.8 | 3.04 | 41.8 | 0.68 | 142 |
| Trend profile + equity/bond premium | 23.0% | 12.6% | 1.83 | -9.5% | 51.8 | 3.17 | 41.8 | 1.32 | 129 |
| Trend profile, learned weights | 20.8% | 12.9% | 1.62 | -10.3% | 67.8 | 3.17 | 41.8 | 1.66 | 161 |
| **KV1** | **15.3%** | **11.9%** | **1.28** | **-19.9%** | **15.3** | **3.57** | **40.4** | **0.93** | **60** |
| **KV2** | **16.6%** | **12.4%** | **1.34** | **-21.8%** | **12.7** | **3.36** | **42.0** | **1.52** | **53** |

After costs, out-of-sample:

| Strategy | Ret | Std | SR | Max DD | Turnover | Leverage | #Names | Imbalance | Costs (bps) |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline TSMOM** | **-0.3%** | **12.0%** | **-0.02** | **-39.1%** | **26.7** | **7.72** | **55.4** | **2.36** | **117** |
| Static benchmark | 5.3% | 15.0% | 0.36 | -29.5% | 2.0 | 4.50 | 55.4 | 4.50 | 41 |
| TSMOM 12m, monthly | -0.8% | 11.1% | -0.07 | -37.1% | 13.8 | 4.50 | 55.4 | 0.66 | 64 |
| Blend 1/3/6/12m | -4.9% | 10.5% | -0.46 | -45.7% | 34.1 | 3.56 | 45.5 | 0.58 | 97 |
| Vol target 11% | -0.5% | 12.4% | -0.04 | -37.3% | 20.8 | 5.13 | 55.4 | 0.80 | 84 |
| Vol target 11%, EWMA risk | 0.6% | 12.2% | 0.05 | -33.4% | 34.4 | 9.01 | 55.4 | 2.62 | 143 |
| Daily EWMA sizing (MOP 2012) | 0.3% | 11.3% | 0.03 | -27.2% | 36.1 | 10.19 | 55.4 | 3.85 | 156 |
| Blend + EWMA vol target | -4.1% | 11.0% | -0.37 | -43.0% | 76.5 | 7.80 | 45.5 | 2.46 | 217 |
| Asset-class risk parity (EWMA) | 0.6% | 12.1% | 0.05 | -35.8% | 56.9 | 11.55 | 55.4 | 2.51 | 208 |
| EWMA vol target + floor | 0.1% | 11.4% | 0.01 | -34.7% | 16.7 | 3.92 | 55.4 | 0.46 | 66 |
| Daily 252d TSMOM, inverse vol | -0.1% | 2.1% | -0.07 | -4.9% | 13.0 | 1.00 | 55.4 | 0.20 | 34 |
| Daily TSMOM 11% vol (floor, buffer) | 0.2% | 12.6% | 0.02 | -29.6% | 55.0 | 4.01 | 55.4 | 0.45 | 143 |
| Daily 252d TSMOM, YZ + corr adjusted | 0.1% | 2.4% | 0.06 | -6.3% | 16.5 | 1.00 | 55.4 | 0.24 | 41 |
| Mix 50/50: EWMA + floor and daily 11% vol | 0.1% | 11.5% | 0.01 | -31.2% | 37.8 | 3.70 | 55.4 | 0.46 | 106 |
| Trend profile | 0.2% | 11.8% | 0.02 | -24.9% | 71.9 | 3.22 | 55.4 | 0.40 | 170 |
| Trend profile + equity/bond premium | 1.3% | 11.2% | 0.12 | -21.7% | 62.4 | 3.40 | 55.4 | 1.47 | 153 |
| Trend profile, learned weights | 1.0% | 10.9% | 0.09 | -23.0% | 60.8 | 3.62 | 55.4 | 2.54 | 151 |
| **KV1** | **1.6%** | **11.8%** | **0.14** | **-30.4%** | **17.5** | **3.98** | **52.8** | **0.55** | **68** |
| **KV2** | **2.8%** | **11.5%** | **0.24** | **-25.0%** | **14.7** | **3.62** | **55.4** | **1.45** | **59** |

### Other strategies

The rest are the strategies from the midterm report (Section 10), the daily TSMOM example with and without correlation-adjusted sizing, and the extensions built before KV1 and KV2.

- **Daily 252d TSMOM, YZ + corr adjusted** is the daily example with Yang-Zhang vol and `CorrelationAdjustedPCM` instead of plain inverse vol, so a contract that moves with many others gets a smaller weight. Both daily examples run at 1x gross, so their returns and vol are small; compare Sharpe ratios.
- **Daily TSMOM 11% vol (floor, buffer)** is the daily example sized with `CorrelationAdjustedPCM` and scaled to 11% ex-ante vol, with the 3% vol floor and a 20% trade buffer. Both parameters were fixed before testing.
- **EWMA vol target + floor** is the midterm's best risk model (EWMA covariance of monthly returns) with the same 3% floor.
- **Mix 50/50** holds half of each of the two above: monthly and daily trend, rebalanced at different speeds.
- **Trend profile** keeps the sizing of the daily 11% vol strategy and changes the signal: the last 12 months as four pieces (months 1, 2-3, 4-6, 7-12), each a z-score capped at 2, averaged, minus the z-score of months 13-15, the reversal after a year in Moskowitz, Ooi and Pedersen (2012). No contract goes above 100% of capital (`PositionCapPCM`).
- **Trend profile + equity/bond premium** adds 0.25 to the signal of every equity and bond future. This part is not trend.
- **Trend profile, learned weights** re-estimates the weight of each of the five pieces, and a lean per asset class, every January by least squares on past data only. It is a check on hindsight, not a strategy to prefer.
- Tested and left out: HRP, risk caps, leverage caps and the continuous signal.

### Caveats

No look-ahead was found: targets from day `t`'s close trade at day `t+1`'s close, the truncation test (`test_no_lookahead`) passes for every strategy, and Sharpe ratios fall slowly with extra delay (above). With each day's returns flipped in sign at random the trend profile earns nothing (`test_trend_profile_earns_nothing_once_returns_are_scrambled`).

1. **Every trend strategy is much weaker after 2014.** In-sample Sharpe ratios of 0.75-1.83 after costs became -0.46 to 0.24 in 2015-2025, and the static long benchmark (0.36) beat them all. The biggest drops were in the strategies that trade most: the trend profile (59-72 times a year) went from 1.73 to 0.02 and the daily 11% vol strategy (50-55) from 1.38 to 0.02.
2. **Concentration.** 1/vol sizing without a floor puts a lot into low-vol contracts. In the baseline, Eurodollar and short sterling are a third of gross exposure and Eurodollar alone reaches 6.4x capital; gross leverage averages 4.9x in-sample and 7.7x out-of-sample. Strategies with the floor stay near 3-4x.
3. **Flat trading cost.** 2 bps per unit traded is applied to every contract and every year back to 1976. High-turnover strategies are the most exposed. KV1 and KV2 trade 13-18 times a year.
4. **No financing.** A futures return is already an excess return, so no financing is charged. A charge on gross leverage above 1x would hit the most leveraged strategies hardest (the baseline above all).
5. **The trend profile was designed on 1976-2014.** Its pieces, the cap and the fade were fixed before the engine run, but the fade was chosen after looking at the 1976-2014 lag profile. The learned-weights version (1.62 in-sample, 0.09 out-of-sample after costs) is the fairer number.

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
- This trades about one day later than the usual convention (trading at the same month-end close as the signal). That convention needs to know a day is the month's last before the next day arrives.

## Daily strategy

Use `DailySchedule`. Same alpha rules. Symbols with no bar today keep their last signal and weight, so there is nothing to merge by hand.

## Execution and costs

- Targets from day `t` are traded `exec_lag` bars later. With `exec_lag=1` they first earn day `t+1`'s return; `main.py` uses `exec_lag=2`.
- Between rebalances positions are held, not rebalanced. Weights drift with prices.
- `main.py` charges 2 bps per unit of traded weight and a roll cost of 2 bps x 4 rolls a year on gross exposure, with no financing charge: a futures return is already an excess return, so holding a leveraged book costs nothing extra. `FuturesCostModel` can also charge financing on gross leverage above a threshold; its defaults (1.5 bps, 1.5 bps x 4 rolls, 50 bps/yr above 1x) are what the tests use.
- A held symbol with no bar earns 0 that day. The first day back earns the whole gap.

## Other notes

- `ModularStrategy` and any alpha holding indicators keep state. Create new instances for every backtest.
- Symbols with too little history (NaN vol) get no weight.
- No margin modeling: positions sized like equities, shorting allowed
- No rollover modeling: assume data has been rolled over already
- Keep it simple: add a new model class only when a real strategy needs one
- Lookahead is designed to be structurally impossible, not just documented as forbidden
