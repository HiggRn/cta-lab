from pathlib import Path

import numpy as np
import pandas as pd

from engine import run_backtest
from panel import Panel
from strategy import MonthlyReturnValidationStrategy


def compare_monthly_returns(
    calculated: pd.DataFrame,
    ground_truth: pd.DataFrame,
    tolerance: float = 1e-4,
) -> None:
    calc = calculated.copy()
    gt = ground_truth.copy()

    # collapse to Period('M') so month-end date labels don't have to line up exactly
    calc.index = calc.index.to_period("M")
    gt.index = gt.index.to_period("M")
    calc = calc[~calc.index.duplicated(keep="last")]
    gt = gt[~gt.index.duplicated(keep="last")]

    calc.columns = [str(c).strip().upper() for c in calc.columns]
    gt.columns = [str(c).strip().upper() for c in gt.columns]

    periods = calc.index.intersection(gt.index)
    symbols = sorted(set(calc.columns) & set(gt.columns))

    print(f"calc: {len(calc)} months x {len(calc.columns)} symbols")
    print(f"gt:   {len(gt)} months x {len(gt.columns)} symbols")
    print(f"overlap: {len(periods)} months x {len(symbols)} symbols")

    if not len(periods) or not symbols:
        print("no overlap between calculated and ground truth")
        return

    if len(gt) > len(calc):
        print("missing month: ", gt.index.difference(calc.index))

    sub_calc = calc.loc[periods, symbols].to_numpy(dtype=np.float64)
    sub_gt = gt.loc[periods, symbols].to_numpy(dtype=np.float64)
    mask = ~np.isnan(sub_calc) & ~np.isnan(sub_gt)
    n = int(mask.sum())
    if n == 0:
        print("no overlapping non-null values")
        return

    diff = np.abs(sub_calc - sub_gt)
    max_diff = float(np.nanmax(diff[mask]))
    mean_diff = float(np.nanmean(diff[mask]))
    corr = float(np.corrcoef(sub_calc[mask], sub_gt[mask])[0, 1])
    matches = int((diff[mask] <= tolerance).sum())

    print(f"match rate (tol={tolerance}): {matches / n:.2%} ({matches}/{n})")
    print(f"max abs diff:  {max_diff:.6e}")
    print(f"mean abs diff: {mean_diff:.6e}")
    print(f"corr: {corr:.6f}")

    if max_diff <= tolerance:
        print("matches MonthlyReturns.csv")
        return

    diff_df = pd.DataFrame(diff, index=periods, columns=symbols)
    top = diff_df.stack().sort_values(ascending=False).head(5)
    print("worst offenders:")
    for (p, s), d in top.items():
        print(
            f"  {p} {s}: calc={calc.loc[p, s]:.6f} gt={gt.loc[p, s]:.6f} diff={d:.6e}"
        )


def main():
    raw_dir = Path("./data/raw")
    asset_map_path = Path("./data/processed/AssetMapCsv.csv")
    monthly_csv_path = Path("./data/processed/MonthlyReturns.csv")

    panel = Panel(
        raw_dirs=raw_dir,
        asset_map_path=asset_map_path,
        dropped_symbols={"EN", "ES", "ER", "SC", "BC", "BG", "ZH", "FF"},
    )

    strategy = MonthlyReturnValidationStrategy()
    run_backtest(panel=panel, strategy=strategy, cost_bps=0.0, exec_lag=1)

    calculated_df = pd.DataFrame.from_dict(
        strategy.recorded_monthly_returns, orient="index"
    ).sort_index()

    gt_df = pd.read_csv(monthly_csv_path, index_col=0, parse_dates=True)
    compare_monthly_returns(calculated_df, gt_df)


if __name__ == "__main__":
    main()
