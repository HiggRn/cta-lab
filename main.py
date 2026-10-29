from pathlib import Path

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
from panel import Panel
from strategy import USER_STRATEGIES

RAW_DIR = Path("./data/raw")
ASSET_MAP_PATH = Path("./data/processed/AssetMapCsv.csv")
SG_TREND_PATH = Path("./data/processed/Lecture3_livedata.xlsx")
FACTOR_DIR = Path("./data/external")
OUTPUT_DIR = Path("./outputs")

# duplicates: ES, SC -> SP and EN -> ND; YM duplicates ZD; bad data: BC, BG, ZH, FF.
# RL stops in 2008, so we keep its mini contract ER instead (dropping both left no Russell 2000 at all).
DROPPED_SYMBOLS = {"EN", "ES", "SC", "BC", "BG", "ZH", "FF", "RL", "YM"}
SAMPLE_START = "1976-02"

BASELINE_NAME = "TSMOM 12m (baseline)"
STATIC_NAME = "Static benchmark"

OUTPUT_DIR.mkdir(exist_ok=True)
panel = Panel(
    raw_dirs=RAW_DIR, asset_map_path=ASSET_MAP_PATH, dropped_symbols=DROPPED_SYMBOLS
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
    name: run_backtest(panel=panel, strategy=cls(), cost_model=cost_model, exec_lag=1)
    for name, cls in all_strategies.items()
}
monthly = {
    name: monthly_returns(res).loc[SAMPLE_START:] for name, res in results.items()
}

# Strategy Performance & Cost Breakdown Table
print(f"\n== Strategy Comparison (from {SAMPLE_START}) ==")
summary = strategy_summary_table(
    results, baseline_name=BASELINE_NAME, sample_start=SAMPLE_START
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
print(correlation_structure(panel, results[target_name]).round(3).to_string())

# Live SG Trend Index Comparison
sg = sg_trend_comparison(target_ret, SG_TREND_PATH)
print(
    f"\n== SG Trend Comparison ({sg['period']}) ==\n"
    f"Corr: {sg['corr']:.2f} | Beta: {sg['beta']:.2f} | Alpha: {sg['alpha']:.1%} (t={sg['t_alpha']:.1f}) | "
    f"SR({target_name}): {sg['SR Strategy']:.2f} vs SR(SG-RF): {sg['SR SG-RF']:.2f}"
)

# Plot Equity & Drawdown Curves
plot_strategies({k: results[k] for k in selected}, sample_start=SAMPLE_START)
