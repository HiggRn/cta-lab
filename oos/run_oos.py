"""Out-of-sample run (parked). From the repo root: `uv run python -m oos.run_oos`

Runs every strategy straight through the join of data/raw and data/oos, then scores
2015-01 onward. See oos/README.md for the known problems before trusting the numbers.
"""

from pathlib import Path

import pandas as pd

from analysis import alpha_table, load_benchmarks, monthly_returns, strategy_summary_table
from baselines import BASELINE_STRATEGIES
from engine import FuturesCostModel, run_backtest
from main import (
    ASSET_MAP_PATH,
    BASELINE_NAME,
    DROPPED_SYMBOLS,
    FACTOR_DIR,
    OUTPUT_DIR,
    RAW_DIR,
    STATIC_NAME,
)
from oos.splice import check_splice
from panel import Panel
from strategy import USER_STRATEGIES

OOS_DIR = Path("./data/oos")
OOS_START = "2015-01"


def main() -> None:
    if not (OOS_DIR.is_dir() and any(OOS_DIR.glob("*.csv"))):
        raise SystemExit(f"No out-of-sample csv files in {OOS_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    splice = check_splice(RAW_DIR, OOS_DIR, DROPPED_SYMBOLS)
    splice.to_csv(OUTPUT_DIR / "oos_splice_check.csv")
    print(f"\n== Out-of-sample join check ({OOS_DIR}) ==")
    print(splice["status"].value_counts().to_string())
    flagged = splice[~splice["status"].isin(["OK", "ENDED"])]
    if not flagged.empty:
        print("Check these symbols before trusting out-of-sample results:")
        print(flagged.round(4).to_string())

    panel = Panel(
        raw_dirs=[RAW_DIR, OOS_DIR],
        asset_map_path=ASSET_MAP_PATH,
        dropped_symbols=DROPPED_SYMBOLS,
    )
    cost_model = FuturesCostModel()
    results = {
        name: run_backtest(panel=panel, strategy=cls(), cost_model=cost_model, exec_lag=2)
        for name, cls in {**BASELINE_STRATEGIES, **USER_STRATEGIES}.items()
    }

    oos_monthly = {
        name: monthly_returns(res).loc[OOS_START:] for name, res in results.items()
    }
    oos_end = str(oos_monthly[BASELINE_NAME].index[-1])
    print(f"\n== Out-of-sample Strategy Comparison ({OOS_START} to {oos_end}) ==")
    summary = strategy_summary_table(
        results, baseline_name=BASELINE_NAME, sample_start=OOS_START
    )
    summary.to_csv(OUTPUT_DIR / "strategy_comparison_oos.csv")
    print(summary.round(3).to_string())

    target = oos_monthly[BASELINE_NAME]
    if len(target) >= 24:  # too few months for a 7-factor regression otherwise
        benchmarks, _ = load_benchmarks(panel, oos_monthly[STATIC_NAME], FACTOR_DIR)
        print(f"\n== Out-of-sample Alpha Attribution for [{BASELINE_NAME}] ==")
        print(alpha_table(target, benchmarks)[["alpha", "t_alpha", "R2"]].round(3).to_string())


if __name__ == "__main__":
    main()
