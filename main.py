from pathlib import Path

import pandas as pd

from analysis import (
    alpha_table,
    correlation_structure,
    load_benchmarks,
    monthly_returns,
    plot_strategies,
    regime_returns,
    sg_trend_comparison,
    smile_regression,
    strategy_summary_table,
)
from baselines import BASELINE_STRATEGIES
from engine import FuturesCostModel, run_backtest
from oos import check_splice
from panel import Panel
from strategy import USER_STRATEGIES

RAW_DIR = Path("./data/raw")
# Held-out data from the course, same file format as data/raw. While this folder is empty
# everything runs in-sample only; once filled, the backtest runs straight through the join.
OOS_DIR = Path("./data/oos")
ASSET_MAP_PATH = Path("./data/processed/AssetMapCsv.csv")
SG_TREND_PATH = Path("./data/processed/Lecture3_livedata.xlsx")
FACTOR_DIR = Path("./data/external")
OUTPUT_DIR = Path("./outputs")

# duplicates: ES, SC -> SP and EN -> ND; YM duplicates ZD; bad data: BC, BG, ZH, FF.
# RL stops in 2008, so we keep its mini contract ER instead (dropping both left no Russell 2000 at all).
DROPPED_SYMBOLS = {"EN", "ES", "SC", "BC", "BG", "ZH", "FF", "RL", "YM"}
SAMPLE_START = "1976-02"
IN_SAMPLE_END = "2014-12"
OOS_START = "2015-01"

BASELINE_NAME = "TSMOM 12m (baseline)"
STATIC_NAME = "Static benchmark"

OUTPUT_DIR.mkdir(exist_ok=True)

# Out-of-sample data: check every file continues smoothly from its last in-sample close
has_oos = OOS_DIR.is_dir() and any(OOS_DIR.glob("*.csv"))
if has_oos:
    splice = check_splice(RAW_DIR, OOS_DIR, DROPPED_SYMBOLS)
    splice.to_csv(OUTPUT_DIR / "oos_splice_check.csv")
    print(f"\n== Out-of-sample join check ({OOS_DIR}) ==")
    print(splice["status"].value_counts().to_string())
    flagged = splice[~splice["status"].isin(["OK", "ENDED"])]
    if not flagged.empty:
        print("Check these symbols before trusting out-of-sample results:")
        print(flagged.round(4).to_string())

panel = Panel(
    raw_dirs=[RAW_DIR, OOS_DIR] if has_oos else RAW_DIR,
    asset_map_path=ASSET_MAP_PATH,
    dropped_symbols=DROPPED_SYMBOLS,
)

cost_model = FuturesCostModel(
    trading_cost_bps=1.5,
    roll_cost_bps=1.5,
    rolls_per_year=4.0,
    financing_cost_bps=50.0,
    free_leverage_threshold=1.0,
)

# Combine course baselines with user custom strategies
all_strategies = {**BASELINE_STRATEGIES, **USER_STRATEGIES}
results = {
    name: run_backtest(panel=panel, strategy=cls(), cost_model=cost_model, exec_lag=2)
    for name, cls in all_strategies.items()
}
# Common comparison window: the latest first full month across all strategies
common_start = max(
    [pd.Period(SAMPLE_START, "M")]
    + [monthly_returns(res).index[0] for res in results.values()]
)
START = str(common_start)
# Everything below up to the out-of-sample section is in-sample only
monthly = {
    name: monthly_returns(res).loc[START:IN_SAMPLE_END] for name, res in results.items()
}

# Strategy Performance & Cost Breakdown Table
print(f"\n== Strategy Comparison ({START} to {IN_SAMPLE_END}) ==")
summary = strategy_summary_table(
    results, baseline_name=BASELINE_NAME, sample_start=START, sample_end=IN_SAMPLE_END
)
summary.to_csv(OUTPUT_DIR / "strategy_comparison.csv")
print(summary.round(3).to_string())

# Factor & Benchmark Alpha Attribution
target_name = BASELINE_NAME
target_ret = monthly[target_name]
benchmarks, factors = load_benchmarks(panel, monthly[STATIC_NAME], FACTOR_DIR)

print(f"\n== Alpha Attribution for [{target_name}] ==")
print(
    alpha_table(target_ret, benchmarks)[["alpha", "t_alpha", "R2"]].round(3).to_string()
)

# Crisis Alpha: Market Smile & Regime Returns
fit_smile = smile_regression(target_ret, factors["Mkt-RF"])
print(
    f"\n== Market Smile for [{target_name}]: beta(Mkt^2) = {fit_smile['beta_Mkt2']:.2f} (t = {fit_smile['t_Mkt2']:.2f}) =="
)
print("\n== Annualized Return by Equity Market Regime ==")
selected = {
    BASELINE_NAME: monthly[BASELINE_NAME],
    STATIC_NAME: monthly[STATIC_NAME],
    **{k: monthly[k] for k in USER_STRATEGIES},
}
print(regime_returns(selected, factors["Mkt-RF"]).round(3).to_string())

# Within vs. Across Asset Class Correlation Structure
print(f"\n== Correlation Structure [{target_name} vs. Passive] ==")
print(
    correlation_structure(panel, results[target_name], end=IN_SAMPLE_END)
    .round(3)
    .to_string()
)

# Live SG Trend Index Comparison
sg = sg_trend_comparison(target_ret, SG_TREND_PATH)
print(
    f"\n== SG Trend Comparison ({sg['period']}) ==\n"
    f"Corr: {sg['corr']:.2f} | Beta: {sg['beta']:.2f} | Alpha: {sg['alpha']:.1%} (t={sg['t_alpha']:.1f}) | "
    f"SR({target_name}): {sg['SR Strategy']:.2f} vs SR(SG-RF): {sg['SR SG-RF']:.2f}"
)

# Out-of-sample: same backtests, signals warmed up on in-sample history, scored from OOS_START
if has_oos:
    oos_monthly = {name: monthly_returns(res).loc[OOS_START:] for name, res in results.items()}
    oos_end = str(oos_monthly[BASELINE_NAME].index[-1])
    print(f"\n== Out-of-sample Strategy Comparison ({OOS_START} to {oos_end}) ==")
    oos_summary = strategy_summary_table(
        results, baseline_name=BASELINE_NAME, sample_start=OOS_START
    )
    oos_summary.to_csv(OUTPUT_DIR / "strategy_comparison_oos.csv")
    print(oos_summary.round(3).to_string())

    print("\n== Net Sharpe Ratio: In-sample vs. Out-of-sample ==")
    print(
        pd.DataFrame(
            {"In-sample": summary["Net SR"], "Out-of-sample": oos_summary["Net SR"]}
        )
        .round(2)
        .to_string()
    )

    oos_target = oos_monthly[target_name]
    if len(oos_target) >= 24:  # too few months for a 7-factor regression otherwise
        oos_benchmarks, _ = load_benchmarks(panel, oos_monthly[STATIC_NAME], FACTOR_DIR)
        print(f"\n== Out-of-sample Alpha Attribution for [{target_name}] ==")
        print(
            alpha_table(oos_target, oos_benchmarks)[["alpha", "t_alpha", "R2"]]
            .round(3)
            .to_string()
        )

# Plot Equity & Drawdown Curves (runs straight through the out-of-sample period if loaded)
plot_strategies({k: results[k] for k in selected}, sample_start=START)
