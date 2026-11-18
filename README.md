# CTA Lab

Multi-asset CTA trend-following backtest system.

## Layout

```
cta-lab/
├── README.md
├── pyproject.toml            # dependencies (managed with uv)
├── uv.lock
├── .python-version
│
├── common.py                 # core interfaces and shared models
├── config.py                 # data paths, sample window, universe cleaning
├── panel.py                  # Panel: raw data -> daily timeline
├── engine.py                 # run_backtest()
├── baselines.py              # baseline strategies and their models
├── strategy.py               # team strategies (incl. KV1, KV2)
├── main.py                   # entry point: runs everything, writes outputs/
├── analysis.py               # performance stats, regressions, studies
│
├── indicators/
│   ├── __init__.py
│   ├── monthly_returns.py
│   └── trend_tstat.py
│
├── scripts/                  # one-off checks, not used by main.py
│   ├── audit_data.py
│   ├── eda.py
│   ├── oos_gap.py
│   └── oos_lecture.py
│
└── tests/
    ├── conftest.py
    ├── test_analysis.py
    ├── test_backtest.py
    ├── test_data.py
    ├── test_engine.py
    ├── test_enhancements.py
    ├── test_models.py
    └── test_panel.py
```

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
- `data/processed/MonthlyReturns.csv` (course file, used by `scripts/eda.py`) and `MonthlyReturns_FullSample.xlsx` (the same to May-2025, used by `scripts/oos_lecture.py`)
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

## Strategy

`ModularStrategy` chains three models:

1. `AlphaModel.update(bars, context)` returns signals. Runs on every bar.
2. `RiskModel.covariance(returns)` returns the annualized covariance of the last `risk_lookback` returns (all history if `None`). Runs on rebalance days only.
3. `PortfolioConstructionModel.create_targets(signals, cov)` returns target weights. Runs on rebalance days only.

The `schedule` passed to `ModularStrategy` decides when to rebalance (`DailySchedule`, `WeeklySchedule`, `FirstAllTradableDayMonthlySchedule` (default) or `TrancheSchedule`).

Current models (`common.py` unless noted):

| Kind | Model | Description |
|---|---|---|
| Alpha | `MonthlyTsmomAlpha` (baselines) | sign of the summed completed monthly returns. Default lookback `(12,)`; several can be averaged, e.g. `(1, 3, 6, 12)` |
| Alpha | `StaticLongAlpha` (baselines) | static benchmark: always long, with the same history requirement as TSMOM |
| Alpha | `NoShortAlpha` (strategy) | wraps an alpha and sets short signals to 0 in equity index futures |
| Alpha | `TrendStrengthAlpha` (strategy) | t-stat of the 252-bar return (log return over its own daily vol x sqrt(252)), capped at +-1 |
| Risk | `SampleCovarianceRiskModel` | covariance of the last `risk_lookback` daily returns |
| Risk | `ExpandingMonthlyRiskModel` (baselines) | covariance of all past monthly returns, at least 36 months. Use `risk_lookback=None` |
| Risk | `VolFloorRiskModel` (strategy) | wraps a risk model and raises any vol below 3%/yr to 3%, keeping correlations |
| Risk | `EwmaVolRiskModel` (strategy) | sample correlations of the last `risk_lookback` daily returns, with EWMA vols (centre of mass 60 days) on the diagonal |
| PCM | `InverseVolPCM` | weight = signal / vol, scaled so gross exposure equals `target_gross_leverage` |
| PCM | `EqualRiskPCM` | signal x 40% / vol, then 1/N; flat until 10 symbols are live |
| PCM | `VolTargetPCM` | wraps another PCM and scales its weights so sqrt(w' cov w) equals `target_vol` |
| PCM | `TranchePCM` (strategy) | wraps a PCM and holds the average of its last four targets, i.e. four tranches of a quarter each |
| Schedule | `TrancheSchedule` (strategy) | rebalances on days 0, 5, 10 and 15 of every 21-trading-day cycle, one tranche at a time |

## Running, metrics, and ouput Logistics
One run of every strategy over the full sample (`uv run main.py`): `data/full_raw`, `exec_lag=2`, 2 bps per unit traded plus 2 bps x 4 rolls a year, no financing.

In-sample data horizon is Jun-1976 to Dec-2014. Out-of-sample data horizon is from Jan-2015 to Jun-2025.

Returns, volatility and Sharpe ratios are annualised from monthly returns, and max drawdown is on compounded wealth. 

Turnover and costs are per year. Leverage is average gross exposure, #Names the average number of positions and Imbalance the average net exposure, in units of capital. `main.py` saves the same numbers to `outputs/strategy_comparison_is.csv` and `outputs/strategy_comparison_oos.csv`.

## Results

### Baseline

`Baseline TSMOM` (`BaselineTsmomStrategy`)

On the first day of each month on which every active contract has traded, each contract gets the sign of its last 12 completed monthly returns, times 40% over its vol (expanding covariance of monthly returns, at least 36 months), then 1/N. No vol targeting. It stays flat until 10 contracts are live. The `Static benchmark` (`StaticBenchmarkStrategy`) uses the same sizing and entry but is always long, no shorting.

### Final strategy: KV

`KV` (`KVStrategy`) starts from 12-month TSMOM rebuilt every 21 trading days, sized 1/252-day vol and scaled to 11% ex-ante vol (step 0, not `Baseline TSMOM`), and adds five changes. Each row adds one change to the row above. Sharpe ratios are after costs; leverage and the largest single position (in units of capital) are in-sample / out-of-sample.

| Step | Change | SR in-sample | SR out-of-sample | Leverage | Largest position |
|---|---|---|---|---|---|
| 0 | 21-day TSMOM, 1/vol, 11% vol | 1.09 | -0.02 | 4.9 / 7.7 | 6.1 / 6.4 |
| 1 | No shorts in equity index futures | 1.11 | 0.10 | 5.0 / 7.8 | 6.1 / 6.4 |
| 2 | EWMA vol, 12-month correlations | 1.22 | 0.13 | 6.9 / 11.7 | 23.7 / 15.8 |
| 3 | 3% vol floor | 1.25 | 0.09 | 3.8 / 4.3 | 0.8 / 0.5 |
| 4 | Four tranches | 1.28 | 0.14 | 3.6 / 4.0 | 0.8 / 0.5 |
| 5 | Trend strength instead of the sign = **KV** | **1.27** | **0.12** | 3.3 / 3.6 | 0.9 / 0.5 |

1. **No shorts in equity index futures** (`NoShortAlpha`). Stock indices drift up and bounce hard after selloffs, so the short side of the 12-month signal mostly loses there. Equity longs stay, and bonds, commodities and currencies can still go short.
2. **Faster risk estimates** (`EwmaVolRiskModel`). Each contract's vol is an EWMA of daily returns (centre of mass 60 days, as in Moskowitz, Ooi and Pedersen 2012) instead of the 252-day vol, and the 11% target uses correlations from the last 12 months instead of all history. Most of the step is the 12-month window (1.19 / 0.18 on its own); EWMA vol on top adds 0.03 in-sample and costs 0.05 out-of-sample.
3. **3% vol floor** (`VolFloorRiskModel`). With faster vol, 1/vol sizing piles into Eurodollar and short sterling: 6.9x gross on average and up to 24x capital in one contract. The floor brings that to 3.8x and under 1x for about the same Sharpe.
4. **Four tranches** (`TrancheSchedule`, `TranchePCM`). The book is split into four parts with a quarter of the capital each, each rebuilt every 21 trading days, a week after the one before. With one book a month the result depends a lot on which day of the month it trades.
5. **Trend strength instead of the sign** (`TrendStrengthAlpha`): the 12-month t-stat capped at +-1, so a weak or noisy trend gets a smaller position. It lowers leverage and turnover but adds nothing to Sharpe.

- Against `Baseline TSMOM` (0.87 / -0.07) the in-sample gain is clear (p = 0.0001), the out-of-sample one is not (p = 0.27). Ten years is short, and 2015-2025 was a bad decade for trend.
- Out-of-sample KV beats the baseline but not the static long benchmark (0.36).
- With every cost doubled: KV 1.23 / 0.07, baseline 0.84 / -0.13.
- Trading later: at `exec_lag` 1, 5, 10 and 21, KV makes 1.30, 1.22, 1.12 and 0.94 in-sample (0.11, 0.10, 0.11, 0.17 out-of-sample).
- KV is net long about 1.0x capital in-sample and 0.6x out-of-sample. The largest single position is 0.91x capital in-sample and 0.54x out-of-sample.
- Passes `test_no_lookahead`.

By sub-period:

| SR after costs | 1976-89 | 1990-99 | 2000-07 | 2008-14 | 2015-19 | 2020-25 |
|---|---|---|---|---|---|---|
| Baseline TSMOM | 1.09 | 1.10 | 1.11 | 0.30 | -0.40 | 0.15 |
| KV | 1.34 | 1.47 | 1.35 | 0.80 | -0.04 | 0.28 |

### All strategies

Before costs, in-sample (Jun-1976 to Dec-2014):

| Strategy | Ret | Std | SR | Max DD | Turnover | Leverage | #Names | Imbalance |
|---|---|---|---|---|---|---|---|---|
| **Baseline TSMOM** | **10.2%** | **11.2%** | **0.91** | **-28.7%** | **8.7** | **3.26** | **39.8** | **0.90** |
| Static benchmark | 6.7% | 14.4% | 0.46 | -67.8% | 1.7 | 3.27 | 39.8 | 3.27 |
| **KV** | **16.6%** | **12.6%** | **1.32** | **-22.9%** | **14.1** | **3.26** | **40.4** | **0.98** |

Before costs, out-of-sample (Jan-2015 to Jun-2025):

| Strategy | Ret | Std | SR | Max DD | Turnover | Leverage | #Names | Imbalance |
|---|---|---|---|---|---|---|---|---|
| **Baseline TSMOM** | **-0.1%** | **11.1%** | **-0.01** | **-35.0%** | **13.8** | **4.50** | **55.4** | **0.66** |
| Static benchmark | 5.7% | 15.0% | 0.38 | -29.2% | 2.0 | 4.50 | 55.4 | 4.50 |
| **KV** | **2.1%** | **11.7%** | **0.18** | **-29.3%** | **16.5** | **3.55** | **52.8** | **0.59** |

After costs, in-sample:

| Strategy | Ret | Std | SR | Max DD | Turnover | Leverage | #Names | Imbalance | Costs (bps) |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline TSMOM** | **9.8%** | **11.2%** | **0.87** | **-28.9%** | **8.7** | **3.26** | **39.8** | **0.90** | **44** |
| Static benchmark | 6.4% | 14.4% | 0.44 | -68.0% | 1.7 | 3.27 | 39.8 | 3.27 | 30 |
| **KV** | **16.0%** | **12.6%** | **1.27** | **-23.0%** | **14.1** | **3.26** | **40.4** | **0.98** | **55** |

After costs, out-of-sample:

| Strategy | Ret | Std | SR | Max DD | Turnover | Leverage | #Names | Imbalance | Costs (bps) |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline TSMOM** | **-0.8%** | **11.1%** | **-0.07** | **-37.1%** | **13.8** | **4.50** | **55.4** | **0.66** | **64** |
| Static benchmark | 5.3% | 15.0% | 0.36 | -29.5% | 2.0 | 4.50 | 55.4 | 4.50 | 41 |
| **KV** | **1.5%** | **11.7%** | **0.12** | **-30.8%** | **16.5** | **3.55** | **52.8** | **0.59** | **62** |

### Caveats

No look-ahead was found: targets from day `t`'s close trade at day `t+1`'s close, the truncation test (`test_no_lookahead`) passes for `Baseline TSMOM` and `KV`, and Sharpe ratios fall slowly with extra delay (above).

1. **Trend is much weaker after 2014.** After costs, KV went from 1.27 in-sample to 0.12 out-of-sample and the baseline from 0.87 to -0.07. The static long benchmark (0.36) beat both in 2015-2025.
2. **Concentration.** 1/vol sizing tilts toward quiet rate futures. In the baseline, Eurodollar, short sterling and 2-year notes are 28% of gross exposure in-sample; out-of-sample, with the Schatz, those four are 42%. No single position goes above 1x capital (baseline 0.7x / 0.8x, KV 0.9x / 0.5x).
3. **Flat trading cost.** 2 bps per unit traded is applied to every contract and every year back to 1976. High-turnover strategies are the most exposed. KV trades 14-17 times a year, the baseline 9-14.
4. **No financing.** A futures return is already an excess return, so no financing is charged. A charge on gross leverage above 1x would hit all three strategies (3.3x in-sample), and the baseline and the static benchmark hardest out-of-sample (4.5x against 3.6x for KV).

## Writing an alpha

- `bars` only contains the symbols that traded today.
- `update()` is called on every bar. Keep one indicator per symbol and feed it every bar you receive.
- An `Indicator` is updated only on days its symbol traded, so "last N bars" means that symbol's own N trading days. Read `.value` only when `.is_ready`.
- Return only the symbols with a new signal. Symbols you leave out keep their last signal. Return `0.0` to go flat.
- `context.get_returns(n)` gives the last `n` daily returns up to today across all symbols (all history if `n` is `None`); `context.get_closes(n)` does the same for closes. NaN means the symbol had no bar that day, and the return over a gap lands on the first day back. pandas reductions (`sum`, `prod`, `std`) skip NaN; plain numpy does not.
- Don't shift anything for execution lag.

## Monthly strategy

Use `FirstAllTradableDayMonthlySchedule`. A month's end is only known once the next month's first bar arrives, so the strategy rebalances on the first day of the new month on which every active symbol has traded (waiting at most 5 days), using completed months only.

- A symbol with no bar on that day still gets a weight, from its last signal.
- This trades about one day later than the usual convention (trading at the same month-end close as the signal). That convention needs to know a day is the month's last before the next day arrives.

## Daily strategy

Use `DailySchedule`. Same alpha rules. Symbols with no bar today keep their last signal and weight, so there is nothing to merge by hand.

## Execution and Transaction Costs

- Targets from day `t` are traded `exec_lag` bars later. With `exec_lag=1` they first earn day `t+1`'s return; `main.py` uses `exec_lag=2`.
- Between rebalances positions are held, not rebalanced. Weights drift with prices.
- `main.py` charges 2 bps per unit of traded weight and a roll cost of 2 bps x 4 rolls a year on gross exposure, with no financing charge: a futures return is already an excess return, so holding a leveraged book costs nothing extra. `FuturesCostModel` can also charge financing on gross leverage above a threshold; its defaults (1.5 bps, 1.5 bps x 4 rolls, 50 bps/yr above 1x) are what the tests use.
- A held symbol with no bar earns 0 that day. The first day back earns the whole gap.
- An order for a symbol with no bar that day waits until the symbol next trades.
- A contract past its last bar is closed at its last price as a normal trade, with turnover and cost charged.

## Other notes

- `ModularStrategy`, any alpha holding indicators, the schedules and `TranchePCM` keep state. Create new instances for every backtest.
- Symbols with too little history (NaN vol) get no weight.
- No margin modeling: positions sized like equities, shorting allowed
- No rollover modeling: assume data has been rolled over already (rolls are only a flat cost)
- Keep it simple: add a new model class only when a real strategy needs one
- Lookahead is designed to be structurally impossible, not just documented as forbidden
