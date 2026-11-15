# CTA Lab

Multi-asset CTA trend-following backtest system.

## Layout

- `common.py`: core interfaces: `Bar`, `Indicator`, `HistoryContext`, `AlphaModel`, `RiskModel`, `PortfolioConstructionModel` (PCM), `Strategy`
- `indicators/`: reusable indicators (`Momentum`, `MonthlyReturn`, `YangZhangVolatility`, `EwmaVolatility`, `TrendProfile`, `TrendTStat`), imported as `from indicators import ...`
- `panel.py`: `Panel`, loads raw data + asset map into the daily timeline
- `engine.py`: `run_backtest()`, runs a `Strategy` over a `Panel`
- `baselines.py`: the course baselines and benchmarks (`BASELINE_STRATEGIES`) and their models
- `strategy.py`: the team's strategies (`USER_STRATEGIES`): the daily TSMOM example, the same example with Yang-Zhang risk and correlation-adjusted sizing, asset-class risk parity from the midterm report, the daily TSMOM at 11% vol, the EWMA vol target with a vol floor, a 50/50 mix of the two, the trend profile strategies (our best in-sample results after costs) and KV1 / KV2
- `main.py`: the entry point. Wires data paths, cost model and params, runs every baseline and user strategy and calls the analysis
- `oos/`: out-of-sample code, parked for now (see `oos/README.md`). Not used by `main.py` or the tests in `tests/`
- `analysis.py`: performance statistics, regressions, benchmark and SG Trend comparison, risk model check, lookback study, lag profile, asset-class and autocorrelation statistics
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
| Schedule | `TrancheSchedule` (strategy) | rebalances on days 0, 5, 10 and 15 of every 21-trading-day cycle, one tranche at a time |
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
| Daily 252d TSMOM, YZ + corr adjusted | 1.26 | 1.17 | -7% | 0.58 | 33 | 0 |
| Asset-class risk parity (EWMA) | 1.07 | 0.83 | -18% | 0.53 | 282 | 213 |
| Daily TSMOM 11% vol (floor, buffer) | 1.49 | 1.30 | -18% | 0.85 | 241 | 143 |
| EWMA vol target + floor | 1.14 | 1.01 | -16% | 0.69 | 143 | 108 |
| Mix 50/50: EWMA + floor and daily 11% vol | 1.42 | 1.25 | -14% | 0.97 | 185 | 116 |
| **Trend profile** | **1.85** | **1.68** | **-10%** | **2.27** | **211** | **104** |
| Trend profile + equity/bond premium | 1.94 | 1.77 | -10% | 2.28 | 207 | 110 |
| Trend profile, learned weights | 1.75 | 1.57 | -11% | 1.90 | 232 | 110 |

- The strategies are the ones discussed in the midterm report (`Summary_Report_Team_1.pdf`, Section 10), the daily TSMOM example with and without correlation-adjusted sizing, and three extensions that do best after costs, each with one simple reason. The report's continuous signal was dropped there and is not included.
- **Daily 252d TSMOM, YZ + corr adjusted** is the daily example with two changes: volatility comes from the Yang-Zhang estimator, and sizing is `CorrelationAdjustedPCM` instead of plain inverse vol, so a contract that is highly correlated with many others gets a smaller weight. Net Sharpe rises from 1.03 to 1.17 (p = 0.02) at slightly higher turnover (18 against 14 a year). It runs at about 1x gross and 3.4% vol.
- **Daily TSMOM 11% vol (floor, buffer)** is the best of the 252-day sign strategies after costs: net Sharpe 1.30 against 0.95 for the best course baseline ("Vol target 11%, EWMA risk", p = 0.002). It is the daily example sized with `CorrelationAdjustedPCM` (inverse vol before; net Sharpe was 1.20) and scaled to 11% ex-ante vol, with a 3% vol floor (stops leverage piling into near-zero-vol contracts) and a 20% trade buffer (roughly halves daily turnover). Both parameters were fixed before testing.
  - Net Sharpe by sub-period: 1.55 (1976-89), 1.23 (1990-99), 1.19 (2000-07) and 0.99 (2008-14), where the baseline drops to 0.20. Among the strategies above it is the best in the first and last periods and within 0.03 of the best (the 50/50 mix) in the other two.
  - With every cost doubled it still has a net Sharpe of 1.10, against 0.74 for the best course baseline.
  - Each layer earns its place: without the floor, net Sharpe is 1.09 (with the buffer only); without the buffer, 1.26 at 1.3x the costs. Leverage caps added nothing on top and were dropped.
- **EWMA vol target + floor** is the report's best risk model (EWMA covariance) with the same 3% vol floor. The floor cuts its costs from 229 to 143 bps/yr and lifts net Sharpe from 0.95 to 1.01 (p < 0.001), and from 0.74 to 0.88 with costs doubled.
- **Mix 50/50** holds half of each: monthly and daily trend, rebalanced at different speeds, much as CTAs blend horizons (report Section 6). Net Sharpe is 1.25 with the smallest drawdown of the three (-14%) and the highest Calmar (0.97). The two sleeves correlate 0.78, so the gain over the daily strategy alone is in drawdown and in relying less on daily trading (turnover 32 against 50 a year), not in Sharpe.
- **Trend profile** keeps the sizing of the daily 11% vol strategy and changes only the signal. The 12-month sign treats every month of the window alike and ignores what came before it. The lag profile printed by `main.py` says the months are not alike: traded alone, each of the last 12 months predicts continuation (gross Sharpe 0.2 to 1.0) and the three months before that predict a reversal (-0.4, -0.3, -0.1), the reversal after a year in Moskowitz, Ooi and Pedersen (2012). So the signal takes the usual 1/3/6/12-month lookbacks as four non-overlapping pieces (months 1, 2-3, 4-6, 7-12), scores each as a z-score capped at 2, averages them and subtracts the z-score of months 13-15. A trend that was already running more than a year ago is old and gets a smaller position. Net Sharpe is 1.68 against 1.30 (p < 0.001), max drawdown -10% against -18%, and no calendar year loses money. `main.py` reports the alpha attribution, market smile, correlation structure and SG Trend comparison for this strategy.
  - Step by step: the four pieces alone give 1.48, and fading months 13-15 adds 0.20. The fade works through commodities (1.24 to 1.51 on commodities alone, 0.96 to 0.91 on the other three classes), where a price move brings a supply response about a crop year later.
  - Net Sharpe by sub-period: 2.05 (1976-89), 1.70 (1990-99), 1.47 (2000-07) and 1.16 (2008-14).
  - With every cost doubled net Sharpe is 1.52. With only the trading cost at 3x, 5x and 10x it is 1.54, 1.40 and 1.05. At `exec_lag` 1, 5, 10 and 21 it is 1.77, 1.61, 1.38 and 1.20.
  - The one sizing change is `PositionCapPCM`: no contract above 100% of capital. It costs nothing in Sharpe, and the largest position is 101% of capital against 208% (caveat 2).
- **Trend profile + equity/bond premium** adds 0.25 to the signal of every equity and bond future, the two classes that pay a premium for being held. This part is not trend. It is about 8% of the P&L and lifts net Sharpe to 1.77 (p = 0.015 against the plain version), with a beta to US equities of -0.03. The same lean on commodities and currencies as well lowers it (1.70).
- **Trend profile, learned weights** is a check on hindsight, not a strategy to prefer. Every January it re-estimates the weight of each of the five pieces, and a lean per asset class, by least squares on the data up to then, so nothing is chosen by hand. Net Sharpe is 1.57. At all 40 refits (1975 to 2014) the four recent pieces get a positive weight and months 13-15 a negative one; the 2014 weights are 15, 11, 7, 17 and -12 (x 0.001 per unit of z-score), with leans of 27 for bonds, 14 for equities, 9 for commodities and 7 for currencies.
- Variants tested and left out: HRP (complex, skew down to -2.4), risk caps (+0.02 at most), leverage caps (nothing on top of the floor) and the continuous signal (lower Sharpe).
- Financing (50 bps/yr on gross leverage above 1x) is 60-80% of costs for every leveraged strategy, so the gap between gross and net Sharpe mostly tracks gross leverage.
- Asset-class risk parity gives fixed income a full quarter of the risk. Bonds have low volatility, so that needs a lot of notional, and its costs are the second highest (282 bps/yr). Net of costs it trails plain EWMA vol targeting (0.83 against 0.95).
- Trading one day sooner (`exec_lag=1`) adds only 0.02-0.07 in-sample (0.04-0.07 for the three daily TSMOM strategies).

### Caveats

No look-ahead was found: signals from day `t`'s close trade at day `t+1`'s close, the truncation test (`test_no_lookahead`) passes for every strategy, the price files are ratio-adjusted with complete OHLC, and net Sharpe decays slowly with extra delay (the daily 11% vol strategy: 1.30 at `exec_lag=2`, 1.33 at 5, 1.14 at 10, 0.97 at 21). With each day's returns flipped in sign at random the trend profile earns nothing (`test_trend_profile_earns_nothing_once_returns_are_scrambled`). The numbers above are still in-sample and should be read with four caveats.

1. **Every choice was made on 1976-2014.** A check on the parked 2015-2024 data (`data/oos`, close-only and noisier, without AP, LB and SS, which have no data after 2014; in-sample Sharpes on those 54 contracts differ slightly from the table above):

   | Net SR | 1976-2014 | 2015-2024 |
   |---|---|---|
   | TSMOM 12m (baseline) | 0.74 | 0.06 |
   | Vol target 11%, EWMA risk | 0.93 | 0.14 |
   | Daily 252d TSMOM, inverse vol | 1.02 | 0.30 |
   | Daily 252d TSMOM, YZ + corr adjusted | 1.14 | 0.54 |
   | Daily TSMOM 11% vol (floor, buffer), with inverse-vol sizing | 1.17 | 0.32 |
   | Daily TSMOM 11% vol (floor, buffer) | 1.25 | 0.44 |
   | Mix 50/50: EWMA + floor and daily 11% vol | 1.20 | 0.33 |

   The ranking holds, and correlation-adjusted sizing beats inverse vol in both periods. The level does not: about 1.3 in-sample became about 0.4 afterwards. The baseline fell by as much, so this is mostly a weak decade for trend following. Without OHLC data the Yang-Zhang estimate equals close-to-close volatility in 2015-2024.
2. **Concentration.** The daily 11% vol strategy averages 3.8x gross leverage (6.9x at most) and has no position cap. Correlation-adjusted sizing favours contracts that move alone, so one contract exceeds 100% of capital on 314 of 9,893 days (FX and US bonds, mostly while few contracts were live) and reached 208% (British pound, June 1977, 19 live contracts). With inverse-vol sizing the largest was 88%.
3. **Flat trading cost.** 1.5 bps per unit traded is applied to every contract and every year back to 1976, and the daily 11% vol strategy turns over 50 times a year. Its net Sharpe is 1.30 at this cost, 1.17 at 3x, 1.05 at 5x and 0.73 at 10x (trading cost only).
4. **The trend profile was designed on this sample.** Its pieces, the cap and the fade were fixed before the engine run, but the fade was chosen after looking at the 1976-2014 lag profile. The learned-weights run (1.57) is the fairer in-sample number. Splitting the 12 months into pieces only helped before 1995: against the daily 11% vol strategy the four pieces make 1.76 against 1.34 in 1976-94 and 1.20 against 1.25 in 1995-2014, so after 1995 the gain is the fade (1.43) and the premium (1.57). In 2005-2014 the trend profile makes 1.01 (1.16 with the premium, 0.93 for the daily 11% vol strategy), so expect the kind of drop seen in caveat 1. It has not been run on `data/oos` yet.

### KV1 and KV2

Two versions of the plain 12-month TSMOM book with a few simple changes. The plain book is the one from the lectures: sign of the 12-month return, size 1/vol, scale the book to 11% ex-ante vol, rebuild it every 21 trading days.

KV1 changes four things:

1. **No shorts in equity index futures.** Stock indices drift up over time and bounce hard after selloffs, so the short side of the 12-month signal mostly loses there. Equity longs stay, and bonds, commodities and currencies can still go short (`NoShortAlpha`).
2. **Faster vol.** Each contract's vol is an EWMA of daily returns with a 60-day centre of mass, as in Moskowitz, Ooi and Pedersen (2012), instead of the 12-month vol. Correlations still come from the last 12 months (`EwmaVolRiskModel`).
3. **3% vol floor**, the same one as in the daily 11% vol strategy (`VolFloorRiskModel`). Without it 1/vol sizing puts about half the gross exposure into Eurodollar and short sterling, more than 20x capital in Eurodollar alone, and financing costs about 3% a year in-sample and over 5% after 2015.
4. **Four tranches.** The book is split into four parts with a quarter of the capital each, and each part is rebuilt every 21 trading days, a week after the one before (`TrancheSchedule`, `TranchePCM`). With one book a month the result depends a lot on which day of the month it trades.

KV2 is KV1 plus two more:

5. **Trend strength instead of the sign.** The signal is the 12-month t-stat capped at +-1 (`TrendStrengthAlpha`). A clear trend still gets a full position, a weak or noisy one gets a smaller one.
6. **Risk-parity sleeve.** Lecture 11 lists risk parity as an extra source of core returns. A quarter of the risk goes to a long-only book of every equity and bond future, sized 1/vol with equal risk for the two classes (`RiskParitySleevePCM`). This part is not trend following: it earns the equity and bond premium, which helps in the years trend does badly.

These numbers are from the current `main.py` setup, not the one behind the table above: one run over the full sample (`data/full_raw`, 1969 to Jun-2025) with 2 bps per unit traded, 2 bps x 4 rolls a year, 50 bps a year financing on gross leverage above 1x and `exec_lag=2`. In-sample is Jun-1976 to Dec-2014 and out-of-sample Jan-2015 to Jun-2025. Returns, vol and Sharpe ratios are annualised from monthly returns, before (gross) and after (net) costs. Vol is the same either way to one decimal. Turnover and costs are per year, and leverage is average gross exposure. The other rows were rerun the same way.

In-sample, Jun-1976 to Dec-2014:

| Strategy | Gross ret | Net ret | Vol | Gross SR | Net SR | Max DD | Turnover | Leverage | Costs (bps) |
|---|---|---|---|---|---|---|---|---|---|
| TSMOM 12m (baseline) | 10.2% | 8.6% | 11.2% | 0.91 | 0.77 | -30% | 8.7 | 3.3 | 159 |
| Daily TSMOM 11% vol (floor, buffer) | 18.2% | 15.4% | 12.2% | 1.49 | 1.27 | -19% | 49.8 | 3.8 | 273 |
| Trend profile | 23.9% | 21.4% | 13.0% | 1.84 | 1.65 | -10% | 58.8 | 3.0 | 246 |
| **KV1** | **15.9%** | **13.9%** | **11.9%** | **1.33** | **1.17** | **-20%** | **15.3** | **3.6** | **190** |
| **KV2** | **17.1%** | **15.4%** | **12.4%** | **1.38** | **1.24** | **-22%** | **12.7** | **3.4** | **173** |

Out-of-sample, Jan-2015 to Jun-2025:

| Strategy | Gross ret | Net ret | Vol | Gross SR | Net SR | Max DD | Turnover | Leverage | Costs (bps) |
|---|---|---|---|---|---|---|---|---|---|
| TSMOM 12m (baseline) | -0.1% | -2.6% | 11.1% | -0.01 | -0.23 | -43% | 13.8 | 4.5 | 244 |
| Daily TSMOM 11% vol (floor, buffer) | 1.7% | -1.3% | 12.6% | 0.13 | -0.10 | -35% | 55.0 | 4.0 | 297 |
| Trend profile | 1.9% | -0.9% | 11.8% | 0.16 | -0.08 | -30% | 71.9 | 3.2 | 284 |
| **KV1** | **2.3%** | **0.1%** | **11.8%** | **0.20** | **0.01** | **-35%** | **17.5** | **4.0** | **220** |
| **KV2** | **3.4%** | **1.5%** | **11.5%** | **0.30** | **0.13** | **-29%** | **14.7** | **3.6** | **193** |

One change at a time, net SR:

| Step | 1976-2014 | 2015-2025 |
|---|---|---|
| Plain book (sign, 12-month vol, one book every 21 days) | 0.96 | -0.27 |
| + no equity shorts | 0.97 | -0.20 |
| + EWMA vol | 0.99 | -0.26 |
| + 3% vol floor | 1.14 | -0.04 |
| + four tranches = **KV1** | 1.17 | 0.01 |
| + trend strength | 1.18 | 0.01 |
| + risk-parity sleeve = **KV2** | 1.24 | 0.13 |

- KV2 has the best 2015-2025 net Sharpe of the trend strategies in the repo. Next are the daily YZ + corr adjusted example (0.06, at 1x gross) and the trend profile with the equity/bond premium (0.01). The static long benchmark did better than all of them (0.24). In-sample the trend profile is still well ahead (1.65 against 1.24), so the two are good at different things.
- Against the baseline the in-sample gain is clear (p < 0.01 for both), the out-of-sample one is not: p = 0.20 for KV1 and 0.11 for KV2. Ten years is short, and 2015-2025 was a bad decade for trend.
- Net SR by sub-period: KV1 1.27 (1976-89), 1.43 (1990-99), 1.17 (2000-07), 0.67 (2008-14), -0.13 (2015-19), 0.15 (2020-25); KV2 1.30, 1.44, 1.25, 0.86, 0.07, 0.18. The baseline: 1.05, 0.94, 0.95, 0.18, -0.60, 0.02.
- Turnover is low (13-18 a year against 50-72 for the daily 11% vol and trend profile strategies), so trading costs are small. Financing is still most of the cost: 120-155 bps a year at 3.4-4x gross. Without it (trading and roll costs only) net SR is 1.28 / 0.14 for KV1 and 1.34 / 0.24 for KV2, against 0.87 / -0.07 for the baseline.
- With every cost doubled: KV1 1.01 / -0.18, KV2 1.10 / -0.04.
- Biggest helpers: the vol floor (mostly by cutting financing), dropping equity shorts and, out-of-sample, the sleeve. EWMA vol and trend strength add little on their own, and EWMA vol is -0.06 out-of-sample at its step.
- Because of the sleeve KV2 is more net long: its average net exposure is 1.5x capital, against 0.6-0.9x for KV1. No single position goes above 1.35x capital in either.
- Both pass `test_no_lookahead`.

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
